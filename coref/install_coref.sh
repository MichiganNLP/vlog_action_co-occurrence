python3 -m venv ./venv
source venv/bin/activate
pip3 install spacy==2.3.7
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
cd ..
python3 -m spacy download en
