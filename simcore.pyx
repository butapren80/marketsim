import datetime
import numpy as np
from libc.math cimport pow as c_power
from libc.math cimport sqrt as c_sqrt
from libc.stdlib cimport rand, RAND_MAX, srand

cdef double rmax = 1.0*RAND_MAX

cpdef set_rand(int seed):
    srand(seed)

cdef class Agent(object):
    
    cdef public long x
    cdef public long y
    cdef public double total_prob
    #cdef public list last_visit
    cdef public long income
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.total_prob = 0.0
        self.income = 100
        #self.last_visit = []

cdef class Store(object):
    cdef public long gsid
    cdef public long operator_id
    cdef public long x
    cdef public long y
     
    def __init__(self, gsid, operator_id, x, y):
        self.gsid = gsid
        self.operator_id = operator_id
        self.x = x
        self.y = y

cdef class Retailer(object):

    cdef public double share
    cdef public long operator_id
    cdef public long num_stores

    def __init__(self, operator_id, share):
        self.share = share
        self.operator_id = operator_id


cpdef calculate_agents_shortlist_and_prob(agents, stores, long N, long O, double max_r, double r_exp):
    cdef Agent agent
    cdef Store store
    cdef int i, j, k, op, n, num_no_store
    cdef double r
    cdef double RMAX = 10000.0
    cdef double [:] rmins = np.zeros(O, dtype = float)
    cdef long [:,:,:] b_int
    cdef double [:,:,:] b_float
    
    m_int = np.zeros( (N, 3, O), dtype = np.int ) # 3 number of data variables
    m_float = np.zeros( (N, 2, O), dtype = np.float ) # 2 number of data variables
    b_int = m_int
    b_float = m_float
    
    for n, agent in enumerate(agents):
        i = agent.x
        j = agent.y   
        
        for op in range(O):
            rmins[op] = RMAX
            b_int[n][2][op] = -100 # last_visit
        
        for store in stores.values():
            op = store.operator_id
            r = c_sqrt( c_power(i - store.x,2 ) + c_power(j - store.y, 2) )
            if r == 0.0:
                r = 0.1
            if r < rmins[op]:
                b_int[n][0][op] = store.gsid # Gloab Store Id
                b_int[n][1][op] = store.operator_id # Operator Id
                
                b_float[n][0][op] = r # Distance r
                rmins[op] = r
                
            elif r ==  rmins[op]:
                #if np.random.rand() < 0.5 :
                if rand()/rmax < 0.5:
                    b_int[n][0][op] = store.gsid
                    b_int[n][1][op] = store.operator_id
                
                    b_float[n][0][op] = r
    
    num_no_store = 0
    for n, agent in enumerate(agents):
        i = agent.x
        j = agent.y
        _r = m_float[n][0]
        
        rf = _r < max_r
        
        prob = 1.0/np.power(_r, r_exp) * rf
        if prob.sum() == 0.0 :
            num_no_store += 1
            prob = rf.astype(float)
        else:
            prob = prob / prob.sum()
        
        m_float[n][1] = prob # Visit probability
        agent.total_prob = prob.sum()
        #print m_float[n][1]

    print 'Locations without operator:', num_no_store
    return m_int, m_float


    
    


def tick_loop(int T, int N, int O, list agents, long [:,:,:] m_int, double [:,:,:] m_float, 
              long [:,:] collector_penetration, long [:,:] collector_operator, long [:,:] collector_gsid):
    cdef int aid, t, pos, gsid, op
    cdef double [:] prob
    cdef double total_prob, ps
    cdef long [:] gsids, ops, last_visit
    
    cdef long TP = 52
    
    for t in range(T):
        print datetime.datetime.now(), 'tick:', t
        for aid, agent in enumerate(agents):
        
            prob =  m_float[aid][1] #agent.prob
            gsids = m_int[aid][0] #agent.gsid
            ops = m_int[aid][1] #agent.operator_id
            last_visit = m_int[aid][2]
            
            total_prob = agent.total_prob
            pos = -1
            if total_prob > 0:
                p = rand() / rmax
                ps = 0.0
                #pos = np.random.multinomial(1, prob).argmax()
                for pos in range(O):
                    ps += prob[pos]
                    if p < ps:
                        break
                #pos = prob.argmax()
                gsid =  gsids[pos]
                op =  ops[pos]

                collector_operator[op][t] += 1
                collector_gsid[gsid][t] += 1
            for op in range(O):
                if op == pos:
                    last_visit[op] = t
                if t - last_visit[op] < TP:
                    collector_penetration[op][t] += 1
    
	
	