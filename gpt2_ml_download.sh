mkdir -p /home/htxu91/data/gpt2-ml/models/mega

perl 3rd/gdown.pl/gdown.pl https://drive.google.com/open?id=1mT_qCQg4AWnAXTwKfsyyRWCRpgPrBJS3 models/mega/model.ckpt-220000.data-00000-of-00001
wget -q --show-progress https://github.com/imcaspar/gpt2-ml/releases/download/v1.0/model.ckpt-220000.index -P models/mega
wget -q --show-progress https://github.com/imcaspar/gpt2-ml/releases/download/v1.0/model.ckpt-220000.meta -P models/mega
echo 'Download finished.🍺'