import os
import yaml
import numpy as np
import pandas as pd
import datetime
import time


class Agent(object):
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.total_prob = 0.0
        #self.last_visit = []

class Store(object):
     
    def __init__(self, gsid, operator_id, x, y):
        self.gsid = gsid
        self.operator_id = operator_id
        self.x = x
        self.y = y

class Retailer(object):

    def __init__(self, operator_id, share):
        self.share = share
        self.operator_id = operator_id
        self.num_stores = 0

def calculate_agents_shortlist_and_prob(agents, stores, N,  O, max_r, r_exp):
    
    RMAX = 10000.0
    rmins = np.zeros(O, dtype = float).tolist()
    
    m_int = np.zeros( (N, 3, O), dtype = np.int ).tolist() # 3 number of data variables
    m_float = np.zeros( (N, 2, O), dtype = np.float ).tolist()  # 2 number of data variables
    
    
    for n, agent in enumerate(agents):
        i = agent.x
        j = agent.y   
        
        for op in range(O):
            rmins[op] = RMAX
            m_int[n][2][op] = -100 # last_visit
        
        for store in stores.values():
            op = store.operator_id
            r = np.sqrt( np.power(i - store.x,2 ) + np.power(j - store.y, 2) )
            if r == 0.0:
                r = 0.1
            if r < rmins[op]:
                m_int[n][0][op] = store.gsid # Gloab Store Id
                m_int[n][1][op] = store.operator_id # Operator Id
                
                m_float[n][0][op] = r # Distance r
                rmins[op] = r
                
            elif r ==  rmins[op]:
                #if np.random.rand() < 0.5 :
                if np.random.random() < 0.5:
                    m_int[n][0][op] = store.gsid
                    m_int[n][1][op] = store.operator_id
                
                    m_float[n][0][op] = r
    
    num_no_store = 0
    for n, agent in enumerate(agents):
        i = agent.x
        j = agent.y
        _r = np.asarray(m_float[n][0])
        
        rf = _r < max_r
        
        prob = 1.0/np.power(_r, r_exp) * rf
        if prob.sum() == 0.0 :
            num_no_store += 1
            prob = rf.astype(float)
        else:
            prob = prob / prob.sum()
        
        m_float[n][1] = prob.tolist() # Visit probability
        agent.prob = prob
        agent.total_prob = prob.sum()
        #print m_float[n][1]

    print 'Locations without operator:', num_no_store
    return m_int, m_float


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


class Sim():

    def __init__(self, config):

        seed = int(time.time())
        print datetime.datetime.now(), 'Seed: ', seed 
        np.random.seed(seed)
        
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
        
        self.collector_gsid = np.zeros( ( len(self.stores), self.T) , dtype = int).tolist()
        self.collector_operator = np.zeros( (self.O, self.T) , dtype = int).tolist()
        self.collector_penetration = np.zeros((self.O, self.T), dtype = int).tolist()


    def run(self):
        print datetime.datetime.now(), 'Running...'

        #tick_loop(self.T, self.N, self.O, self.agents, self.m_int, self.m_float, 
        #          self.collector_penetration, self.collector_operator, self.collector_gsid)
        
        TP = 52
        
        m_float = self.m_float
        m_int = self.m_int
        collector_operator = self.collector_operator
        collector_gsid = self.collector_gsid
        collector_penetration = self.collector_penetration
        
        
        for t in range(self.T):
            print datetime.datetime.now(), 'tick:', t
            for aid, agent in enumerate(self.agents):
            
                prob =  agent.prob #m_float[aid][1] #agent.prob
                gsids = m_int[aid][0] #agent.gsid
                ops = m_int[aid][1] #agent.operator_id
                last_visit = m_int[aid][2]
                
                total_prob = agent.total_prob
                pos = -1
                if total_prob > 0:
                    p = np.random.random()
                    
                    pos = np.random.multinomial(1, prob).argmax()
                    
                    #pos = prob.argmax()
                    gsid =  gsids[pos]
                    op =  ops[pos]
        
                    collector_operator[op][t] += 1
                    collector_gsid[gsid][t] += 1
                for op in range(self.O):
                    if op == pos:
                        last_visit[op] = t
                    if t - last_visit[op] < TP:
                        collector_penetration[op][t] += 1

    
    def posprocess(self):
        
        print datetime.datetime.now(), 'Postprocessing...'
        store = pd.HDFStore('output.h5')
        
        df_operator = pd.DataFrame(np.asarray(self.collector_operator).T , columns = ['operator_%d' % d for d in range(self.O)])
        store['operators'] = df_operator
        
        gsids = self.stores.keys()
        gsids.sort()
        df_stores = pd.DataFrame(np.asarray(self.collector_gsid).T, columns = ['S%d' % d for d in gsids] )
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
            row.extend( m_int[i][0] )
            row.extend( m_float[i][0] )
            row.extend( m_int[i][1] )
            row.extend( m_float[i][1] )
            
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
        
        df_penetration = pd.DataFrame(np.asarray(self.collector_penetration).T, columns = ['operator_%d' % d for d in range(self.O)] )
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