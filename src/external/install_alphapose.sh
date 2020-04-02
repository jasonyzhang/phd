git clone git@github.com:MVIG-SJTU/AlphaPose.git
cd AlphaPose
git fetch origin 38e00c688023282304462b5b6da98248e798842e  # API tested with this commit.
python setup.py build develop --user
echo ""
echo "Don't forget to download the pre-trained model and configs!"

