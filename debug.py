from model.build_model import Build_Model
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Build_Model(weight_path=None, resume=False, dims=3).to(device)
    inputs = []
    #inputs.append(torch.randn(8,1,128,128,128))
    inputs.append(torch.randn(5,1,160,160,160))
    print("\n"*2)
    for t in inputs:
        t = t.to(device)
        p, p_d = model(t)
        print("t:", t.shape)
        print("p:", [p_scale.shape for p_scale in p])
        print("p_d:", [p_scale.shape for p_scale in p_d])
        print("="*30)


if __name__ == "__main__":
    main()