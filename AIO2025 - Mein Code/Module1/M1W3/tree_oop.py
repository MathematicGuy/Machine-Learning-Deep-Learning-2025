class TreeNode:
    """
        Step 1: Add Node
        Step 2: Organize node

    """
    def __init__(self, name):
        self.name = name
        self.childrens = []
        self.parent = None # a node do or do not have parent node e.g. root node


    def add_childrens(self, child: "TreeNode"):
        child.parent = self # set node's parent to the current node
        self.childrens.append(child) # append children nodes to current node


    def get_depth(self):
        """
        Count the depth of current node by tracing current node parent
        then the parent of their parent, repeat until there are no parent left.
        This indicate we've reached the root node


        Returns:
            depth: int
        """
        depth = 0
        p = self.parent # p mean parrent

        while p : #? if current node have parent then execute below if
            depth += 1
            p = p.parent #? double pointer: replace "current parrent" by parrent's node parrent

        return depth

    def visualize_tree(self):
        space = ' ' * self.get_depth() * 3 + '|_'
        print(space + self.name) # add prefix

        #! This is HUGE: I can access children node inside a node recursively
        if self.childrens:
            #? access children node of the current node starting from the top node
            for child in self.childrens:
                child.visualize_tree() #? Retursively visualize tree in each node, auto return if there childrens is None because of recursive


def build_tree():
    anode = TreeNode("A")
    bnode = TreeNode("B")
    cnode = TreeNode("C")
    dnode = TreeNode("D")
    enode = TreeNode("E")
    fnode = TreeNode("F")
    gnode = TreeNode("G")

    # Level 3 nodes
    hnode = TreeNode("H")
    inode = TreeNode("I")
    jnode = TreeNode("J")
    knode = TreeNode("K")

    anode.add_childrens(bnode)
    anode.add_childrens(cnode)

    bnode.add_childrens(dnode)
    bnode.add_childrens(enode)

    cnode.add_childrens(fnode)
    cnode.add_childrens(gnode)

    # Add level 3 nodes
    dnode.add_childrens(hnode)
    # enode.add_childrens(inode)
    fnode.add_childrens(jnode)
    # gnode.add_childrens(knode)

    return anode

if __name__ == '__main__':

    tree = build_tree()
    tree.visualize_tree()