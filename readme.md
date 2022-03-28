### CKDRO ###
### Environment Setup ###
First, create a conda environment to hold the dependencies.
```
conda create --name DRO python=3.8 -y
source activate DRO
pip install -r requirements.txt
```

### DC OPF example ###
To run the DC OPF example, simply run 
```
python DRO_ERCOT.py
```

### data and network ###
We included the ERCOT network data and the load profile in the data folder, you can add other network models and load data as you need. 

To generate/modify the pickle file from raw data, run 
```
cd data/ERCOT
python load_hourly.py
```