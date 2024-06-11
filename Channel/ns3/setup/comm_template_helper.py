import json
import yaml
import argparse

def func(ops):
    order = ops['order']
    order_json = {}

    for ind, item in enumerate(order):
        try:
            order_json[str(ind+1)] = {"from": str(item),
                                    "to": str(order[ind+1])
                                    }
        except IndexError as e:
            # Last item does not have a "to" node
            None

    with open("../Device/peer/comm_template.json", "w") as outfile:
        json.dump(order_json, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', "-p", required=True, type = str)
    args = parser.parse_args()
    ops = yaml.safe_load(open(args.path, "r"))
    func(ops)