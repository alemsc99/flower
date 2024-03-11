from pathlib import Path
import pickle

import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn

import flwr as fl
from server import get_evaluate_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg:DictConfig):

    # Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # Dataset preparation
    trainloaders, validationloaders, testloaders=prepare_dataset(cfg.num_clients,
                                                                  cfg.batch_size)
    # print(len(trainloaders), len(trainloaders[0].dataset)) # number of clients and samples in the first trainloader

    # Define clients
    client_fn=generate_client_fn(trainloaders, validationloaders, cfg.model)

    #Define strategy
    
    strategy=instantiate(cfg.strategy,evaluate_fn=get_evaluate_fn(cfg.model, testloaders)
                                        )
    # Start simulation
    history=fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': cfg.num_cpus, 'num_gpus': cfg.num_gpus} #'num_gpus' is the ratio of vram the client should 
                                                          # have access to. 1.0 means i can run one client at a
                                                          # time; 0.25, for example, means i can run up to 4 clients
                                                          # Default: 0.0 -> no gpu        
    )

    # Saving results
    save_path=HydraConfig.get().runtime.output_dir
    results_path=Path(save_path)/'results.pkl'

    results={'history':history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    



if __name__=="__main__":
    main()
