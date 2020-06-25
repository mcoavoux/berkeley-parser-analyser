from nlp_util import pstree, render_tree, init, treebanks, parse_errors, head_finder, tree_transform
from collections import defaultdict
from StringIO import StringIO


## Note: those two are duplicated from transform search
def get_label(tree):
    if tree.word is None:
        return tree.label
    if tree.label == 'PU':
        return tree.label + tree.word
    else:
        return tree.label


def get_preterminals(tree, ans=None):
    return_tuple = False
    if ans is None:
        ans = []
        return_tuple = True
    if tree.is_terminal():
        ans.append(tree.label)
    for subtree in tree.subtrees:
        assert subtree != tree
        get_preterminals(subtree, ans)
    if return_tuple:
        return tuple(ans)



def disco_gen_different_label_successor(node, cur_label, new_label):
    success, response = tree_transform.change_label_by_node(node, new_label, in_place=True)
    assert success, response
    ntree, nnode = response

    info = {
        'type': 'relabel',
        'change': (cur_label, new_label),
        'subtrees': [get_label(subtree) for subtree in nnode.subtrees],
        'parent': nnode.parent.label,
        'span': nnode.span,
        'family': [get_label(subtree) for subtree in nnode.parent.subtrees],
        'auto preterminals': get_preterminals(nnode),
        'auto preterminal span': nnode.span,
        'over_word': len(nnode.subtrees) == 1 and nnode.subtrees[0].word is not None
    }
    return (True, ntree, info)


def disco_add_node(tree, span, label, position=0, in_place=True):
    '''Introduce a new node in the tree.  Position indicates what to do when a
    node already exists with the same span.  Zero indicates above any current
    nodes, one indicates beneath the first, and so on.'''
    
    tree = tree.root()
    if not in_place:
        tree = tree.clone()

    # Search node such that: its span has the smallest superset of span
    set_span = set(span)
    node = tree.find_disco_spanning_node(set_span)
    
    # search whether the subtrees have either:
    #    empty intersection with span
    #    or are subsets of span
    outnodes = []
    innodes = []
    for subtree in node.subtrees:
        intersection = subtree.true_span & set_span
        if len(intersection) == 0:
            outnodes.append(subtree)
        elif len(intersection) == len(subtree.true_span):
            innodes.append(subtree)
        else:
            return (False, "Cannot add node, crossing problem")
    
    full_span = set()
    for inode in innodes:
        full_span |= inode.true_span
    assert full_span == set_span

    newnode = pstree.PSTree(word=None, label=label, span=(0,0), parent=node, subtrees=innodes)
    newchildren = outnodes + [newnode]
    node.subtrees = sorted(newchildren, key = lambda x: min(x.true_span))
    
    tree.update_node_order()
    tree.update_proj_spans()
    
    return (True, (tree, newnode))
    
def disco_gen_missing_successor(ctree, error):
    success, response = disco_add_node(ctree, error[1], error[2], in_place=True)
    assert success, response

    ntree, nnode = response
    nnode_index = nnode.parent.subtrees.index(nnode)

    info = {
        'type': 'add',
        'label': get_label(nnode),
        'span': nnode.span,
        'subtrees': [get_label(subtree) for subtree in nnode.subtrees],
        'parent': nnode.parent.label,
        'family': [get_label(subtree) for subtree in nnode.parent.subtrees],
        'auto preterminals': get_preterminals(nnode),
        'auto preterminal span': nnode.span,
        'left siblings': nnode.parent.subtrees[:nnode_index],
        'right siblings': nnode.parent.subtrees[nnode_index + 1:],
        'over_word': len(nnode.subtrees) == 1 and nnode.subtrees[0].is_terminal(),
        'over words': reduce(lambda prev, node: prev and node.is_terminal(), nnode.subtrees, True),
    }

    return (True, ntree, info)


def disco_remove_node_by_node(node, in_place):
    if not in_place:
        node = pstree.clone_and_find(node)
    root = node.root()
    parent = node.parent
    position = parent.subtrees.index(node)
    init_position = position
    parent.subtrees.pop(position)
    for subtree in node.subtrees:
        subtree.parent = parent
        parent.subtrees.append(subtree)
    parent.subtrees.sort(key = lambda x: min(x.true_span))
    parent.update_proj_spans()
    return (True, (parent, node, None, None))



def disco_gen_extra_successor(ctree, error, gold):
    #ans.append(('extra', tspan, tnode.label, tnode))
    #success, response = disco_remove_node(ctree, error[1], error[2], in_place=False)
    success, response = disco_remove_node_by_node(error[3], in_place=True)
    assert success, response

    parent, dnode, spos, epos  = response
    ntree = parent.root()
    ntree.update_node_order()
    ntree.update_proj_spans()
    

    info = {
        'type': 'remove',
        'label': get_label(dnode),
        'span': dnode.span,
        'subtrees': [get_label(subtree) for subtree in dnode.subtrees],
        'parent': parent.label,
        'family': [get_label(subtree) for subtree in parent.subtrees[:spos] + [dnode] + parent.subtrees[epos:]],
        # 'left siblings': [get_label(subtree) for subtree in parent.subtrees[:spos]],
        # 'right siblings': [get_label(subtree) for subtree in parent.subtrees[epos:]],
        'left siblings': [get_label(subtree) for subtree in parent.subtrees], # for error analysis
        'right siblings': [get_label(subtree) for subtree in parent.subtrees],
        'over words': reduce(lambda prev, node: prev and node.is_terminal(), dnode.subtrees, True),
        'over_word': len(dnode.subtrees) == 1 and dnode.subtrees[0].is_terminal(),
        'auto preterminals': get_preterminals(parent),
        'auto preterminal span': parent.span
    }

    """
    # that might be useful
    if len(info['right siblings']) == 1:
        sibling = parent.subtrees[-1]
        for node in sibling:
            if node.word is not None:
                gold_eq = gold.get_nodes('lowest', node.span[0], node.span[1])
                if gold_eq is not None:
                    if get_label(node) != gold_eq.label:
                        info['POS confusion'] = (get_label(node), get_label(gold_eq))
    """
    
    return (True, ntree, info)

def find_possible_new_parent(ancestors, new_span, new_label, true_gold_spans):
    # find all nodes that would be gold with new span
    #  are not necessarily gold now
    # to rank by: lowest + are gold with current span
    new_span = set(new_span)
    res = []
    for node in ancestors:
        future_span_set = new_span | node.true_span
        future_span_tup = tuple(sorted(future_span_set))
        if future_span_tup in true_gold_spans and node.label in true_gold_spans[future_span_tup]:
            
            tspan = tuple(sorted(node.true_span))
            was_gold = tspan in true_gold_spans and node.label in true_gold_spans[tspan]
            
            # Create new const if:
            # candidate parent has a different span or a diferent label
            #   node, was_gold before movement, True if no need to create new node
            res.append((node, was_gold, node.label == new_label and future_span_set == new_span))
    return res

def disco_gen_move_successor(ctree, gold_true_spans, error):
    # arguments: ctree, gspan, gold_true_spans

    root = ctree.root()
    error_type, gspan, glabel, gnode = error
    assert error_type == "crossing"

    children_nodes = find_children_nodes(ctree, gspan)

    # If they have the same parent -> not happening bc wouldn't be crossing
    # for each newchild:
    #    check if parent minus node is still a gold const
    #    detach them from parents
    # create new node

    # search where to attach new node:
    #    lowest node such that a gold constituent is created?
    #    lowest common ancestor -> safest
    # recompute proj spans

    info = []
    children_nodes.sort(key = lambda x: min(x.true_span))


    parents_clone = [c.parent.clone() for c in children_nodes]

    ancestors = []
    for c in children_nodes:
        current = c
        while current is not None:
            ancestors.append(current)
            current = current.parent

    res = find_possible_new_parent(ancestors, gspan, glabel, gold_true_spans)
    # sorting: priority to no new node creation
    res.sort(key = lambda x: (- x[2], x[1], len(x[0].true_span)))
    best_candidate = res[0]
    best_node = best_candidate[0]

    add_and_move = not best_candidate[2]
    if not add_and_move:
        # don't create a new node
        for c in children_nodes:
            if c == best_node:
                print "same node, do nothing"
            elif c.parent != best_node:
                position = c.parent.subtrees.index(c)
                c.parent.subtrees.pop(position)

                best_node.subtrees.append(c)
                best_node.true_span |= c.true_span
                c.parent = best_node
            else:
                print "Same parent"
    else:
        print "add and move"
        for c in children_nodes:
            parent = c.parent
            position = parent.subtrees.index(c)
            parent.subtrees.pop(position)
            parent.true_span -= c.true_span

        new_node = pstree.PSTree(word=None, label=glabel, span=(0,0), parent=None, subtrees=children_nodes)

        # create new node
        best_node.subtrees.append(new_node)
        best_node.true_span |= new_node.true_span
        best_node.subtrees.sort(key=lambda x: min(x.true_span))
        new_node.parent = best_node

    root = best_node.root()
    root.update_true_spans()
    #print "root before reordering", root
    root.update_node_order()
    #print "root after  reordering", root
    root.update_proj_spans()
    

    info = {
        'type': 'move',
        'old_parent': "|".join([get_label(c) for c in parents_clone]),
        'new_parent': glabel,
        'movers': [get_label(node) for node in children_nodes],
        'mover info': [(get_label(node), node.true_span) for node in children_nodes],
        'new_family': [get_label(subtree) for subtree in best_node.subtrees],
        'old_family': [get_label(subtree) for subtree in c for c in parents_clone],
        'start left siblings': [get_label(node) for node in c.subtrees for c in parents_clone],
        'start right siblings': [get_label(node) for node in c.subtrees for c in parents_clone],
        'end left siblings': [get_label(node) for node in c.subtrees for c in parents_clone],
        'end right siblings': [get_label(node) for node in c.subtrees for c in parents_clone],
        'auto preterminals': get_preterminals(best_node),
        'auto preterminal span': best_node.span,
        'added and moved': add_and_move,
        'added label': glabel,
        
        'number_moved': len(children_nodes),
    }

    #print "New tree", root
    return (True, root, info)



def find_children_nodes(tree, span):
    ## return the minimal list of nodes such that their span is span
    if all([i in span for i in tree.true_span]):
        return [tree]
    l = []
    for subtree in tree.subtrees:
        if any([i in span for i in subtree.true_span]):
            l.extend(find_children_nodes(subtree, span))
    return l

def disco_successors(ctree, cerrors, gold):
    # error: ('extra', tspan, tnode.label, tnode))
    # error: (type, span, label, tnode
    # Change the label of a node

    print "Len missing ", len(cerrors.missing)
    print "Len extra   ", len(cerrors.extra)
    print "Len crossing", len(cerrors.crossing)

    made_modifications = False
    for merror in cerrors.missing:
        # ans.append((name, gspan, gnode.label, gnode))
        _, gspan, glabel, gnode = merror
        for eerror in cerrors.extra:
            # error: ans.append(('extra', tspan, tnode.label, tnode))
            _, tspan, tlabel, tnode = eerror
            if gspan == tspan:
                #yield disco_gen_different_label_successor(tnode, merror[1], eerror[1], merror[2])
                print "relabelling", tlabel, glabel
                made_modifications = True
                yield disco_gen_different_label_successor(tnode, tlabel, glabel)
    if made_modifications:
        return
        
    # Add a node
    for error in cerrors.missing:
        print "missing"
        yield disco_gen_missing_successor(ctree, error)

    # Remove a node
    for error in cerrors.extra:
        print "extra"
        yield disco_gen_extra_successor(ctree, error, gold)


    # Iterate on crossing errorss
    # find smallest list of nodes to form missing constituent
    # create node
    # check that no gold constituent is undone
    # where to attach:
    #   parent of one of them if it ends up in correct constituent
    #   lowest common parent
    gold_true_spans = {}
    for node in gold:
        true_span = tuple(sorted(node.true_span))
        if true_span in gold_true_spans:
            gold_true_spans[true_span].append(node.label)
        else:
            gold_true_spans[true_span] = [node.label]


    for crossing in cerrors.crossing:

        #yield gen_move_successor(source_span, left, right, new_parent, cerrors, gold)
        print "crossing", crossing
        yield disco_gen_move_successor(ctree, gold_true_spans, crossing)
        return # 1 crossing at a time -> solving a crossing might change type of other errors and introducing new ones


def greedy_search_disco(gold, test, classify):
    # Initialise with the test tree
    cur = (test.clone(), {'type': 'init'}, 0)
    iters = 0
    path = []
    #print "gold", gold
    #print "pred", test
    while True:
        path.append(cur)
        if iters > 100:
            return (0, iters), None
        # Check for victory

        ctree = cur[0]
        cerrors = parse_errors.ParseErrorSet(gold, ctree, discontinuous=True)
        
        if len(cerrors) == 0:
            final = cur
            break

        best = None
        # search for and apply fixes, returns iterator on fixes, newtree, information
        for fixes, ntree, info in disco_successors(ctree, cerrors, gold):
            #print "ntree", ntree
            if not ntree.check_consistency():
                raise Exception("Inconsistent tree! {}".format(ntree))
            nerrors = parse_errors.get_disco_errors(ntree, gold)
            change = len(cerrors) - len(nerrors)
            # authorize introducing at most one new error
            # motivations: some discontinuous errors cannot be repaired
            # without modifying projective constituents
            if change < -1:
                continue

            if best is None or change > best[2]:
                best = (ntree, info, change)
        cur = best
        iters += 1
    
    # TODO:
    # assert the tokens are the same as in the gold tree
    # make custom classify functions
    
    # print gold
    # print final[0]
    gold_words = gold.word_yield(as_list=True)
    pred_words = final[0].word_yield(as_list=True)
    if gold_words != pred_words:
        print
        print "gold", gold
        print
        for t in path:
            print t[0]
        print gold_words
        print pred_words
        print
        assert False, "Error, bad discontinuous error correction"

    for step in path:
        classify(step[1], gold, test)
        if step[1]["classified_type"] != "UNSET init":
            print step[1]["classified_type"]
    return (0, iters), path
