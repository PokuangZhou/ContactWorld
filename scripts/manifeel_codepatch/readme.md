## some extra folder

```bash
cd ContactWorld
mkdir thirdparty
cd ContactWorld/data
mkdir demo_data
mkdir pretrained_model
cd pretrained_model
mkdir dino3
```

## git clone mafeel repo
```bash
cd ContactWorld/thirdparty
git clone https://github.com/purdue-mars/manifeel.git
```

## then replace this install.sh to manifeel one

ContactWorld/scripts/manifeel_codepatch/install.sh
👉
ContactWorld/thirdparty/manifeel/install.sh

## data and dependncy download 
need download Download the TacSL specific Isaac Gym
download data (Download temporarily from Google Drive)

```bash
pip install gdown
```

IsaacGym rc4
```bash
cd /ContactWorld/thirdparty
gdown "https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing"
tar -xzf IsaacGym_Preview_TacSL_Package.tar.gz
```

dinov3 
```bash
cd ContactWorld/data/pretrained_model/dino3
gdown "https://drive.google.com/file/d/1m_WYeLRM50KT6M2MfTUtJro2px5e0Rmt/view?usp=sharing"
```

data
```bash
cd ContactWorld/data/demo_data
gdown "data\link"
```

install manifeel repo
```bash
cd ContactWorld/thirdparty/manifeel
insntall.sh 
```

add world model code part (this repo is based on manifeel repo)
```bash
cd ContactWorld/scripts/manifeel_codepatch/
run replace_code.sh
```

Markdown
### demo data and checkpoint download link
💾 [huggingface link](https://huggingface.co/datasets/Pokuang/ContactWorld)