import glob,os
import torch
from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO,greedy_action

ckpt=torch.load("weights/ppo_upmsp.pth",map_location="cpu")
model=PPO(ckpt["obs_dim"],ckpt["action_dim"],ckpt["pad_n"],ckpt["pad_m"])
model.load_state_dict(ckpt["model"])
model.eval()

os.makedirs("results",exist_ok=True)
out=open("results/solutions.txt","w",encoding="utf-8")

for file in sorted(glob.glob("instances/*.txt")):

    env=UPMSPEnv(read_instance(file),ckpt["pad_n"],ckpt["pad_m"])
    env.reset()

    while True:
        s=torch.tensor(env.state(),dtype=torch.float32).unsqueeze(0)
        mask=torch.tensor(env.action_mask()).unsqueeze(0)
        action=greedy_action(model,s,mask)

        _,_,done,_,info=env.step(int(action.item()))

        if done:
            text=(f"{os.path.basename(file)}\n"
                  f"Cmax: {info['makespan']:.1f}  (ratio to LB: "
                  f"{info['makespan']/env.scale:.3f})\n"
                  f"solution: {info['solution']}\n")
            print(text.split("\n")[0]); print(text.split("\n")[1])
            out.write(text)
            break

out.close()
print("saved results/solutions.txt")
