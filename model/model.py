import numpy as np
from scipy.special import softmax

NSTATES = 9
MAX_SCH = 205

class SchemaTabularBayes():
    """ CRP prior
    tabluar predictive distirbution
    """

    def __init__(self,concentration,stickiness_wi,stickiness_bt,sparsity,
        lrate=1,lratep=1,pvar=0,decay_rate=1,schidx=None):
        self.Tmat = np.zeros([NSTATES,NSTATES])
        self.alfa = concentration
        self.beta_wi = stickiness_wi
        self.beta_bt = stickiness_bt - np.abs(pvar*np.random.randn(1)[0])
        self.lrate = lrate # lrate like
        self.lratep = lratep # lrate prior
        self.lmbda = sparsity
        self.ntimes_sampled = 0
        self.schidx = schidx
        self.decay_rate = decay_rate

    def get_prior(self,beta_mode,ztm1,ztrm1):
        """ beta_mode controls whether to combine betas or use separate
        ztm1 : z of tstep t minus 1
        ztrm1 : z of trial t minus 1
        """
        if self.ntimes_sampled == 0:
            return self.alfa
        ztm1_flag = ztm1 == self.schidx
        ztrm1_flag = ztrm1 == self.schidx
        if beta_mode == 0: # beta within only
            crp = self.lratep*self.ntimes_sampled + self.beta_wi* ztm1_flag
        elif beta_mode == 1: # beta between only
            assert ztm1 == ztrm1
            crp = self.lratep*self.ntimes_sampled + self.beta_bt* ztm1_flag
        elif beta_mode == 2: # combined
            crp = self.lratep*self.ntimes_sampled + \
                    self.beta_bt*ztrm1_flag + self.beta_wi*ztm1_flag
        return crp

    def get_like(self,xtm1,xt):
        PARAM_S = 2
        num = self.lmbda + self.Tmat[xtm1,xt]
        den = (PARAM_S*self.lmbda) + self.Tmat[xtm1,:].sum()
        like = num/den
        return like

    def update(self,xtm1,xt):
        self.Tmat[xtm1,xt]+=self.lrate
        return None

    def decay(self):
        self.Tmat = self.Tmat*self.decay_rate
        return None

    def predict(self,xtm1):
        """ returns un-normalized count """
        xthat = np.array([
            self.get_like(xtm1,x) for x in range(NSTATES)
            ])
        return xthat


class SEM():

    def __init__(self,schargs,beta2,skipt1,ppd_allsch):
        self.SchClass = SchemaTabularBayes
        self.schargs = schargs
        self.beta2_flag = beta2
        self.skipt1 = skipt1
        self.ppd_allsch = ppd_allsch
        self.init_schlib()

    def init_schlib(self):
        """ 
        initialize with two schemas
        one active one inactive
        """
        sch0 = self.SchClass(**self.schargs,schidx=0)
        sch1 = self.SchClass(**self.schargs,schidx=1)
        self.schlib = [sch0,sch1]
        return None

    def decay_allsch(self):
        for sch in self.schlib:
            sch.decay()
        return None

    def get_beta_mode(self):
        if self.tstep==0: 
            return 1 # between only
        elif self.beta2_flag:
            return 2 # between+within
        else:
            return 0 # within only
        return None

    def calc_posteriors(self,xtm1,xt,ztm,ztrm,active_only=False):
        """ loop over schema library
        """
        beta_mode = self.get_beta_mode()
        if active_only: # prediction
            priors = [sch.get_prior(beta_mode,ztm,ztrm) for sch in self.schlib if sch.ntimes_sampled>0]
            likes = [sch.get_like(xtm1,xt) for sch in self.schlib if sch.ntimes_sampled>0]
            # print(self.tstep,likes)
        else: # sch inference
            priors = [sch.get_prior(beta_mode,ztm,ztrm) for sch in self.schlib]
            likes = [sch.get_like(xtm1,xt) for sch in self.schlib]
            # record 
            self.data['prior'][self.tridx,self.tstep,:len(priors)] = priors
            self.data['like'][self.tridx,self.tstep,:len(likes)] = likes
        posteriors = [p*l for p,l in zip(priors,likes)]
        return posteriors

    def select_sch(self,xtm1,xt,ztm,ztrm):
        """ xt and xtm1 are ints
        """
        posteriors = self.calc_posteriors(xtm1,xt,ztm,ztrm)
        self.data['post'][self.tridx,self.tstep,:len(posteriors)] = posteriors
        active_k = np.argmax(posteriors)
        if active_k == len(self.schlib)-1:
            self.schlib.append(self.SchClass(**self.schargs,schidx=len(self.schlib)))
        return active_k

    def predict(self,xtm1,ztm,ztrm):
        """ 
        """
        pr_xt_z = np.array([
            self.calc_posteriors(xtm1,x,ztm,ztrm,active_only=True
                ) for x in range(NSTATES)
            ]) # probability of each next state under each schema
        pr_xtp1 = np.sum(pr_xt_z,axis=1) # sum over schemas
        return pr_xtp1

    def run_exp(self,exp):
        """ exp is L of trialL
        trialL is L of obs (ints) 
        """
        ## recording
        Mt0ij = np.zeros([len(exp),NSTATES,NSTATES])
        self.data = data = {
            'zt':-np.ones([len(exp),len(exp[0])]),
            'xth':-np.ones([len(exp),len(exp[0]),NSTATES]),
            'prior':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'like':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
            'post':-np.ones([len(exp),len(exp[0]),MAX_SCH]),
        }
        ## 
        scht = schtm = schtrm = self.schlib[0] # sch0 is active to start
        # scht.ntimes_sampled += 1
        for tridx,trialL in enumerate(exp):
            self.tridx = tridx
            for tstep,(xtm,xt) in enumerate(zip(trialL[:-1],trialL[1:])):
                # conditional
                if (tstep==1) and (self.skipt1): 
                    continue
                if len(self.schlib)>=MAX_SCH: return data
                # print('ts',tstep)
                self.tstep = tstep
                ## prediction: marginilize over schemas
                if self.ppd_allsch:
                    xth = self.predict(xtm,schtm.schidx,schtrm.schidx)
                else:
                    xth = scht.predict(xtm)
                ## prediction: only active schema
                # xth = scht.predict(xtm)
                # update infered active schema
                zt = self.select_sch(xtm,xt,schtm.schidx,schtrm.schidx)
                scht = self.schlib[zt]
                # print(tstep,scht.Tmat,self.data['likes'][:20,:,1])
                ## forgetting
                scht.decay()
                # update transition matrix
                scht.update(xtm,xt)
                scht.ntimes_sampled += 1
                # update schema history
                schtm = scht
                data['xth'][tridx][tstep] = xth
                data['zt'][tridx][tstep] = zt       
            # final schema of trial
            schtrm = scht 
        return data




class Task():
    """ 
    """

    def __init__(self):
        A1,A2,B1,B2 = self._init_paths_csw()
        self.paths = [[A1,A2],[B1,B2]]
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None


    def _init_paths_csw(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        begin = 0 # E0
        locA,locB = 1,2 # E1
        node11,node12 = 3,4 # E2 
        node21,node22 = 5,6 # E3
        node31,node32 = 7,8 # E4
        end = 9
        A1 = np.array([begin,locA,
            node11,node21,node31
            ])
        A2 = np.array([begin,locA,
            node12,node22,node32
            ])
        B1 = np.array([begin,locB,
            node11,node22,node31
            ])
        B2 = np.array([begin,locB,
            node12,node21,node32
            ])
        return A1,A2,B1,B2

    def _init_paths_toy(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        locA,locB = 0,1
        node11,node12 = 2,3
        node21,node22 = 4,5
        A1 = np.array([locA,
            node11,node21
            ])
        A2 = np.array([locA,
            node12,node22
            ])
        B1 = np.array([locB,
            node11,node22
            ])
        B2 = np.array([locB,
            node12,node21
            ])
        return A1,A2,B1,B2


    def get_curriculum(self,condition,n_train,n_test):
        """ 
        order of events
        NB blocked: ntrain needs to be divisible by 4
        """
        curriculum = []   
        if condition == 'blocked':
            assert n_train%4==0
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4 )
        elif condition == 'early':
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 4)
        elif condition == 'middle':
            curriculum =  \
                [0, 1] * (n_train // 8) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 8)
        elif condition == 'late':
            curriculum =  \
                [0, 1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4)
        elif condition == 'interleaved':
            curriculum = [0, 1] * (n_train // 2)
        elif condition == 'single': ## DEBUG
            curriculum =  \
                [0] * (n_train) 
        else:
            print('condition not properly specified')
            assert False
        # 
        curriculum += [int(np.random.rand() < 0.5) for _ in range(n_test)]
        return np.array(curriculum)


    def generate_experiment(self,condition,n_train,n_test):
        """ 
        exp: arr [ntrials,tsteps]
        curr: arr [ntrials]
        """
        # get curriculum
        n_trials = n_train+n_test
        curr = self.get_curriculum(condition,n_train,n_test)
        # generate trials
        exp = -np.ones([n_trials,self.tsteps],dtype=int)
        for trial_idx in range(n_train+n_test):
            # select A1,A2,B1,B2
            event_type = curr[trial_idx]
            path_type = np.random.randint(2)
            path_int = self.paths[event_type][path_type]
            # embed
            exp[trial_idx] = path_int
        return exp,curr