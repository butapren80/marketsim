import os
import yaml
import numpy as np
import pandas as pd
import datetime
import time

import pyximport; pyximport.install(setup_args = {'include_dirs' : np.get_include()})
from simcore import * 


def allocateIncome(agent, N):
    x = agent.x
    y = agent.y
    income = int(np.random.normal(y+50, x+10))
    return income
        
    

def random_alloction(N, O, nstores):
    gsid = 0
    stores = {}
    sigs = set()
    for operator in range(O):
        for n in range(nstores[operator]):
            while 1:
                x = np.random.randint(N-10) + 5
                y = np.random.randint(N-10) + 5
                sig = '%d%d' % (x,y)
                if sig not in sigs:
                    break
            sigs.add(sig)
            store = Store(gsid, operator, x, y)
            stores[gsid] = store
            gsid += 1
    return stores

def evolutionary_allocation(N, O, retailers, rmin):
    
    gsid = 0
    stores = {}
    sigs = set()
    ns = {}
    total = 0
    shares = []
    for operator in range(O):
        ns[operator] = 0
        shares.append(retailers[operator].share)
        total += retailers[operator].num_stores
    n = 0
    while 1 :
        # pick operator at random, proportional to its final share
        operator = np.random.multinomial(1, shares).argmax()
        if ns[operator] >= retailers[operator].num_stores:
            continue
        while 1:
            x = np.random.randint(N-10) + 5
            y = np.random.randint(N-10) + 5
            found = 0
            for _x, _y in sigs:
                r = np.sqrt( np.power(_x-x, 2) + np.power(_y-y,2) )
                if r < rmin :
                    found = 1
                    break
            if found == 0:
                sigs.add((x,y))
                store = Store(gsid, operator, x, y)
                stores[gsid] = store
                gsid += 1
                ns[operator] += 1
                n += 1
                #print gsid,
                break
        if n >= total:
            break
    #print
    print '{:<11}'.format('Operator Id'), '{:>7}'.format('Stores')
    print '-'*23
    for operator in range(O):
        print '{:>11}'.format('%d' % operator), '{:>7}'.format('%d' % ns[operator])
    
    return stores

def assing_open_ticks(stores, T):
    
    L = len(stores)
    for store in stores.values():
        store.tick_open = -1


class Sim():

    def __init__(self, config):

        seed = int(time.time())
        print datetime.datetime.now(), 'Seed: ', seed 
        set_rand(seed) # set c random seed
        
        self.N = config['model_parameters']['agents']['N']
        self.O = config['model_parameters']['operators']['N']
        self.shares = config['model_parameters']['operators']['shares']
        self.T = config['model_parameters']['ticks']
        self.S = config['model_parameters']['stores']
        
        self.rmin = config['model_parameters']['operators']['rmin'] # Minimum distance between stores
        self.max_r = config['model_parameters']['agents']['max_r']  # Maximum distance from Agent to Store
        self.r_exp = config['model_parameters']['agents']['r_exp']  # Travel time exponent
        
        

    def preprocess(self):
        print datetime.datetime.now(), 'Preprocessing...'
        
        # Create retailers and calculate their number of initial stores
        self.retailers = {}
        for operator in range(self.O):
            retailer = Retailer(operator, self.shares[operator])
            self.retailers[operator] = retailer
 
        sr = self.S
        #nstores = []
        for operator in range(self.O):
            sn = int(self.S * self.shares[operator])
            self.retailers[operator].num_stores = sn
            #nstores.append(sn)
            sr -= sn
            #print operator, self.retailers[operator].num_stores
        if sr != 0 and operator == self.O - 1:
            print 'Warning, incorrect number of exisitng stores, sr = %d' % sr
    
        #self.nstores = nstores
        nsq = int(np.sqrt(self.N))
        self.stores = evolutionary_allocation(nsq, self.O, self.retailers, self.rmin)
        
        
        # Create agents
        self.agents = []
        n = 0
        for i in range(nsq):
            for j in range(nsq):
                agent = Agent(i, j)
                self.agents.append(agent)
        
        m_int, m_float = calculate_agents_shortlist_and_prob(self.agents, self.stores, self.N, self.O, 
                                            max_r = self.max_r, r_exp = self.r_exp)
        self.m_int = m_int
        self.m_float = m_float
        
        self.collector_gsid = np.zeros( ( len(self.stores), self.T) , dtype = int)
        self.collector_operator = np.zeros( (self.O, self.T) , dtype = int)
        self.collector_penetration = np.zeros((self.O, self.T), dtype = int)


    def run(self):
        print datetime.datetime.now(), 'Running...'

        tick_loop(self.T, self.N, self.O, self.agents, self.m_int, self.m_float, 
                  self.collector_penetration, self.collector_operator, self.collector_gsid)

    
    def posprocess(self):
        
        print datetime.datetime.now(), 'Postprocessing...'
        store = pd.HDFStore('output.h5')
        
        df_operator = pd.DataFrame(self.collector_operator.T , columns = ['operator_%d' % d for d in range(self.O)])
        store['operators'] = df_operator
        
        gsids = self.stores.keys()
        gsids.sort()
        df_stores = pd.DataFrame(self.collector_gsid.T, columns = ['S%d' % d for d in gsids] )
        store['stores'] = df_stores
        
        rows = []
        columns = ['id', 'x', 'y', 'total_prob']
        columns.extend( 'gsid_%d' % d for d in range(self.O) )
        columns.extend( 'r_%d' % d for d in range(self.O) )
        columns.extend( 'operator_id_%d' % d for d in range(self.O) )
        columns.extend( 'prob_%d' % d for d in range(self.O) )
        
        m_int = self.m_int
        m_float = self.m_float
        
        for i, agent in enumerate(self.agents):
            row = [i, agent.x, agent.y, agent.total_prob]
            row.extend( m_int[i][0].tolist() )
            row.extend( m_float[i][0].tolist() )
            row.extend( m_int[i][1].tolist() )
            row.extend( m_float[i][1].tolist() )
            
            #row.extend( agent.gsid.tolist() )
            #row.extend( agent.r.tolist() )
            #row.extend( agent.operator_id.tolist() )
            #row.extend( agent.prob.tolist() )
            
            rows.append( row )
        df_agents = pd.DataFrame.from_records(rows, columns = columns)
        store['agents'] = df_agents
        
        gsids = self.stores.keys()
        gsids.sort()
        rows = []
        for gsid in gsids:
            _store = self.stores[gsid]
            row = [gsid, _store.operator_id, _store.x, _store.y]
            rows.append(row)
        df_stores_data = pd.DataFrame.from_records(rows, columns = ['gsid', 'operator_id', 'x','y'])
        store['stores_data'] = df_stores_data
        
        # TP = 52
        # d = {}
        # num = 0
        # for agent in self.agents:
        #     s = agent.prob.sum()
        #     if s > 0.0 :
        #         num += 1
        # f = 100.0 / num
        
        # for op in range(self.O):
        #     y = []
        #     for t in range(self.T):
        #         n = 0
        #         for agent in self.agents:
        #             if t - agent.penetration[op][t] < TP:
        #                 n += 1
        #         y.append(n * f)
        #     d[op] = y
        # x = [t for t in range(self.T)]
        # df_penetration = pd.DataFrame(np.asarray( [d[op] for op in range(self.O)] ).T, columns = ['operator_%d' % d for d in range(self.O)])
        df_penetration = pd.DataFrame(self.collector_penetration.T, columns = ['operator_%d' % d for d in range(self.O)] )
        store['penetration'] = df_penetration
        
        print datetime.datetime.now(), 'Done...'
        store.close()
        
        
    
        

if __name__ == '__main__':
	
    config = yaml.load(open('config.yaml'))
    
    sim = Sim(config)
    
    sim.preprocess()
    sim.run()
    sim.posprocess()
    
    #print config
    #print 'hello'