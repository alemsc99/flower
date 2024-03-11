import hydra 
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

def function_test(x,y):
    result=x+y
    print(f"{result=}")
    return result

class MyClass:
    def __init__(self, x):
        self.x=x

    def print_x_squared(self):
        print(f"{self.x**2}")

@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print(cfg.foo)
    print(cfg.bar.baz)
    print(cfg.bar.more)
    print(cfg.bar.more.blabla)


    output=call(cfg.my_func)
    print(f"{output=}")

    output=call(cfg.my_func, y=100)
    print(f"{output=}")

    print("partials")
    fn=call(cfg.my_partial_func)
    output=fn(y=100)

    print(f"{output=}")

    print("objects")
    obj:MyClass=instantiate(cfg.my_object)
    obj.print_x_squared()

    print(cfg.toy_model)
    mymodel=instantiate(cfg.toy_model)
    print(mymodel)

if __name__=="__main__":
    main()