import json

order = [2,6,4,1,6,5,2,6,4,1,3,7,2,6,4]
order_json = {}

for ind, item in enumerate(order):
    try:
        order_json[str(ind+1)] = {"from": str(item),
                                  "to": str(order[ind+1])
                                  }
    except IndexError as e:
        # Last item does not have a "to" node
        None

with open("comm_template.json", "w") as outfile:
    json.dump(order_json, outfile)