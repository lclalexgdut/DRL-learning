import numpy as np

def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        lines=[x.strip() for x in f if x.strip()]

    n,m,_=map(int, lines[0].split())

    processing=np.zeros((m,n), dtype=np.float32)

    for j in range(n):
        data=list(map(int, lines[2+j].split()))
        for k in range(0,len(data),2):
            processing[data[k],j]=data[k+1]

    idx=2+n
    assert lines[idx]=="SSD"
    idx+=1

    setup=np.zeros((m,n,n), dtype=np.float32)

    for machine in range(m):
        assert lines[idx]==f"M{machine}"
        idx+=1
        for i in range(n):
            setup[machine,i]=list(map(int, lines[idx].split()))
            idx+=1

    return {
        "n":n,
        "m":m,
        "processing":processing,
        "setup":setup
    }
