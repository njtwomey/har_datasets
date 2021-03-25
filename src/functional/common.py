def sorted_node_values(nodes):
    return [nodes[key] for key in sorted(nodes.keys())]


def node_itemgetter(item):
    def itemgetter_func(df):
        return df[item]

    itemgetter_func.__name__ = f"get_{item}"

    return itemgetter_func
