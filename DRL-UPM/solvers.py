import numpy as np

def decode_batch(perms,p,s):
    # assign jobs in permutation order, each to the machine minimizing the
    # resulting makespan increment (setup-aware). Shared by SPT and GA.
    # perms: [P,n]  p: [m,n]  s: [m,n,n]
    P,n=perms.shape
    m=p.shape[0]
    mt=np.zeros((P,m))
    last=np.full((P,m),-1,dtype=np.int64)
    mstar=np.zeros((P,n),dtype=np.int64)
    ar=np.arange(P)
    for t in range(n):
        jobs=perms[:,t]                                  # [P]
        has=last>=0                                      # [P,m]
        su=s[np.arange(m)[:,None],np.where(has.T,last.T,0),jobs[None,:]]
        cost=p[:,jobs]+su*has.T                          # [m,P]
        finish=mt.T+cost
        cur=mt.max(axis=1)                               # [P]
        score=np.maximum(finish,cur[None,:])*1e6+finish  # tie-break: min finish
        k=score.argmin(axis=0)                           # [P]
        mt[ar,k]=finish[k,ar]
        last[ar,k]=jobs
        mstar[:,t]=k
    return mt.max(axis=1),mstar

def build_solution(perm,mstar,m):
    sol=[[] for _ in range(m)]
    for t,j in enumerate(perm):
        sol[int(mstar[t])].append(int(j))
    return sol

def spt(data):
    # static rule: shortest minimum-processing-time job first
    p,s,n,m=data["processing"],data["setup"],data["n"],data["m"]
    perm=np.argsort(p.min(axis=0))
    cmax,mstar=decode_batch(perm[None],p,s)
    return float(cmax[0]),build_solution(perm,mstar[0],m)

def _ox(p1,p2,rng):
    n=len(p1)
    a,b=sorted(rng.integers(0,n,2))
    child=np.empty(n,dtype=np.int64)
    mid=set(p1[a:b+1].tolist())
    child[a:b+1]=p1[a:b+1]
    fill=[x for x in p2 if x not in mid]
    idx=0
    for i in list(range(b+1,n))+list(range(0,a)):
        child[i]=fill[idx]; idx+=1
    return child

def run_ga(data,seed=0,pop=80,gen=150,elite=4):
    p,s,n,m=data["processing"],data["setup"],data["n"],data["m"]
    rng=np.random.default_rng(seed)
    perms=np.argsort(rng.random((pop,n)),axis=1)
    perms[0]=np.argsort(p.min(axis=0))    # SPT seed
    perms[1]=np.argsort(-p.min(axis=0))   # LPT seed
    fits,_=decode_batch(perms,p,s)
    for g in range(gen):
        cand=rng.integers(0,pop,size=(pop,3))
        winners=cand[np.arange(pop),fits[cand].argmin(axis=1)]
        newp=perms[winners].copy()
        for i in range(0,pop-1,2):
            if rng.random()<0.9:
                c1=_ox(newp[i],newp[i+1],rng)
                c2=_ox(newp[i+1],newp[i],rng)
                newp[i],newp[i+1]=c1,c2
        for i in range(pop):
            if rng.random()<0.3:
                c1,c2=sorted(rng.integers(0,n,2))
                newp[i,c1:c2+1]=newp[i,c1:c2+1][::-1]
        newfits,_=decode_batch(newp,p,s)
        order=np.argsort(fits)
        worst=np.argsort(newfits)[-elite:]
        for e,wi in enumerate(worst):
            newp[wi]=perms[order[e]]
            newfits[wi]=fits[order[e]]
        perms,fits=newp,newfits
    bi=int(fits.argmin())
    cmax,mstar=decode_batch(perms[bi:bi+1],p,s)
    return float(cmax[0]),build_solution(perms[bi],mstar[0],m)
