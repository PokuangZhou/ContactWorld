## git clone manifeel repo, and then replace this install.sh to manifeel one
```bash
cd thirdparty
git clone https://github.com/purdue-mars/manifeel.git
```
when git clone is finished
```bash
cd ..
cp scripts/manifeel_codepatch/install.sh thirdparty/manifeel/install.sh
```

## data and dependncy download 
need download Download the TacSL specific Isaac Gym
download data (Download temporarily from Google Drive)

```bash
pip install gdown
```

IsaacGym rc4
```bash
cd thirdparty
gdown "https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing"
```
when download is finished
```bash
tar -xzf IsaacGym_Preview_TacSL_Package.tar.gz
```

install manifeel repo
```bash
cd manifeel
bash install.sh 
```
add world model code part (this repo is based on manifeel repo)
```bash
cd ../..
cd scripts/manifeel_codepatch/
bash replace_code.sh
```

dinov3 
```bash
cd ContactWorld/data/pretrained_model/dino3
gdown "https://drive.google.com/file/d/1m_WYeLRM50KT6M2MfTUtJro2px5e0Rmt/view?usp=sharing"
```

## demo data and checkpoint download link
💾 [huggingface link](https://huggingface.co/datasets/Pokuang/ContactWorld)