from nlp_util import pstree, render_tree, init, treebanks, parse_errors, head_finder, tree_transform
from collections import defaultdict
from StringIO import StringIO


# def gen_different_label_successor(ctree, span, cur_label, new_label):
    # success, response = tree_transform.change_label(ctree, new_label, span, cur_label, False)
    # assert success, response

    # ntree, nnode = response

    # info = {
        # 'type': 'relabel',
        # 'change': (cur_label, new_label),
        # 'subtrees': [get_label(subtree) for subtree in nnode.subtrees],
        # 'parent': nnode.parent.label,
        # 'span': nnode.span,
        # 'family': [get_label(subtree) for subtree in nnode.parent.subtrees],
        # 'auto preterminals': get_preterminals(nnode),
        # 'auto preterminal span': nnode.span,
        # 'over_word': len(nnode.subtrees) == 1 and nnode.subtrees[0].word is not None
    # }

    # return (True, ntree, info)


# def gen_missing_successor(ctree, error):
    # success, response = tree_transform.add_node(ctree, error[1], error[2], in_place=False)
    # assert success, response

    # ntree, nnode = response
    # nnode_index = nnode.parent.subtrees.index(nnode)

    # info = {
        # 'type': 'add',
        # 'label': get_label(nnode),
        # 'span': nnode.span,
        # 'subtrees': [get_label(subtree) for subtree in nnode.subtrees],
        # 'parent': nnode.parent.label,
        # 'family': [get_label(subtree) for subtree in nnode.parent.subtrees],
        # 'auto preterminals': get_preterminals(nnode),
        # 'auto preterminal span': nnode.span,
        # 'left siblings': nnode.parent.subtrees[:nnode_index],
        # 'right siblings': nnode.parent.subtrees[nnode_index + 1:],
        # 'over_word': len(nnode.subtrees) == 1 and nnode.subtrees[0].is_terminal(),
        # 'over words': reduce(lambda prev, node: prev and node.is_terminal(), nnode.subtrees, True),
    # }

    # return (True, ntree, info)


# def gen_extra_successor(ctree, error, gold):
    # success, response = tree_transform.remove_node(ctree, error[1], error[2], in_place=False)
    # assert success, response

    # parent, dnode, spos, epos  = response
    # ntree = parent.root()

    # info = {
        # 'type': 'remove',
        # 'label': get_label(dnode),
        # 'span': dnode.span,
        # 'subtrees': [get_label(subtree) for subtree in dnode.subtrees],
        # 'parent': parent.label,
        # 'family': [get_label(subtree) for subtree in parent.subtrees[:spos] + [dnode] + parent.subtrees[epos:]],
        # 'left siblings': [get_label(subtree) for subtree in parent.subtrees[:spos]],
        # 'right siblings': [get_label(subtree) for subtree in parent.subtrees[epos:]],
        # 'over words': reduce(lambda prev, node: prev and node.is_terminal(), dnode.subtrees, True),
        # 'over_word': len(dnode.subtrees) == 1 and dnode.subtrees[0].is_terminal(),
        # 'auto preterminals': get_preterminals(parent),
        # 'auto preterminal span': parent.span
    # }

    # if len(info['right siblings']) == 1:
        # sibling = parent.subtrees[-1]
        # for node in sibling:
            # if node.word is not None:
                # gold_eq = gold.get_nodes('lowest', node.span[0], node.span[1])
                # if gold_eq is not None:
                    # if get_label(node) != gold_eq.label:
                        # info['POS confusion'] = (get_label(node), get_label(gold_eq))

    # return (True, ntree, info)


# def gen_move_successor(source_span, left, right, new_parent, cerrors, gold):
    # success, response = tree_transform.move_nodes(source_span.subtrees[left:right+1], new_parent, False)
    # assert success, response

    # ntree, nodes, new_parent = response
    # new_left = new_parent.subtrees.index(nodes[0])
    # new_right = new_parent.subtrees.index(nodes[-1])

    # # Find Lowest Common Ancestor of the new and old parents
    # full_span = (min(source_span.span[0], new_parent.span[0]), max(source_span.span[1], new_parent.span[1]))
    # lca = new_parent
    # while not (lca.span[0] <= full_span[0] and full_span[1] <= lca.span[1]):
        # lca = lca.parent

    # info = {
        # 'type': 'move',
        # 'old_parent': get_label(source_span),
        # 'new_parent': get_label(new_parent),
        # 'movers': [get_label(node) for node in nodes],
        # 'mover info': [(get_label(node), node.span) for node in nodes],
        # 'new_family': [get_label(subtree) for subtree in new_parent.subtrees],
        # 'old_family': [get_label(subtree) for subtree in source_span.subtrees],
        # 'start left siblings': [get_label(node) for node in source_span.subtrees[:left]],
        # 'start right siblings': [get_label(node) for node in source_span.subtrees[right+1:]],
        # 'end left siblings': [get_label(node) for node in new_parent.subtrees[:new_left]],
        # 'end right siblings': [get_label(node) for node in new_parent.subtrees[new_right+1:]],
        # 'auto preterminals': get_preterminals(lca),
        # 'auto preterminal span': lca.span
    # }

    # if left == right and nodes[-1].span[1] - nodes[-1].span[0] == 1:
        # preterminal = nodes[-1]
        # while preterminal.word is None:
            # preterminal = preterminal.subtrees[0]
        # gold_eq = gold.get_nodes('lowest', preterminal.span[0], preterminal.span[1])
        # if gold_eq is not None:
            # info['POS confusion'] = (get_label(preterminal), get_label(gold_eq))

    # # Consider fixing a missing node in the new location as well
    # nerrors = parse_errors.ParseErrorSet(gold, ntree)
    # to_fix = None
    # for error in nerrors.missing:
        # if error[1][0] <= nodes[0].span[0] and nodes[-1].span[1] <= error[1][1]:
            # if error[1] == (nodes[0].span[0], nodes[-1].span[1]):
                # continue
            # if error[1][0] < new_parent.span[0] or error[1][1] > new_parent.span[1]:
                # continue
            # if to_fix is None or to_fix[1][0] < error[1][0] or error[1][1] < to_fix[1][1]:
                # to_fix = error
    # if to_fix is not None:
        # info['added and moved'] = True
        # info['added label'] = error[2]

        # unmoved = []
        # for node in new_parent.subtrees:
            # if to_fix[1][0] < node.span[0] and node.span[1] < to_fix[1][1]:
                # if node not in nodes:
                    # unmoved.append(node)
        # info['adding node already present'] = False
        # if len(unmoved) == 1 and unmoved[0].label == to_fix[2]:
            # info['adding node already present'] = True

        # success, response = tree_transform.add_node(ntree, to_fix[1], to_fix[2], in_place=False)
        # assert success, response
        # ntree, nnode = response

    # return (False, ntree, info)



def disco_successors(ctree, cerrors, gold):
    # error: ('extra', tspan, tnode.label, tnode))
    # error: (type, span, label, tnode
    # Change the label of a node
    for merror in cerrors.missing:
        for eerror in cerrors.extra:
            print(merror, eerror)
            if merror[1] == eerror[1]:
                yield gen_different_label_successor(ctree, eerror[1], eerror[2], merror[2])

    # Add a node
    for error in cerrors.missing:
        yield gen_missing_successor(ctree, error)

    # Remove a node
    for error in cerrors.extra:
        yield gen_extra_successor(ctree, error, gold)

    # Move nodes
    for source_span in ctree:
        # Consider all continuous sets of children
        for left in xrange(len(source_span.subtrees)):
            for right in xrange(left, len(source_span.subtrees)):
                if left == 0 and right == len(source_span.subtrees) - 1:
                    # Note, this means in cases like (NP (NN blah)) we can't move the NN
                    # out, we have to move the NP level.
                    continue
                new_parents = []

                # Consider moving down within this bracket
                if left != 0:
                    new_parent = source_span.subtrees[left-1]
                    while not new_parent.is_terminal():
                        if cerrors.is_extra(new_parent):
                            new_parents.append(new_parent)
                        new_parent = new_parent.subtrees[-1]
                if right != len(source_span.subtrees) - 1:
                    new_parent = source_span.subtrees[right+1]
                    while not new_parent.is_terminal():
                        if cerrors.is_extra(new_parent):
                            new_parents.append(new_parent)
                        new_parent = new_parent.subtrees[0]

                # If source_span is extra
                if cerrors.is_extra(source_span) and (left == 0 or right == len(source_span.subtrees) - 1):
                    # Consider moving this set out to the left
                    if left == 0:
                        if source_span.subtrees[left].span[0] > 0:
                            for new_parent in ctree.get_nodes('all', end=source_span.subtrees[left].span[0]):
                                if cerrors.is_extra(new_parent):
                                    new_parents.append(new_parent)

                    # Consider moving this set out to the right
                    if right == len(source_span.subtrees) - 1:
                        if source_span.subtrees[right].span[1] < ctree.span[1]:
                            for new_parent in ctree.get_nodes('all', start=source_span.subtrees[right].span[1]):
                                if cerrors.is_extra(new_parent):
                                    new_parents.append(new_parent)

                    # Consider moving this set of spans up
                    if left == 0:
                        # Move up while on left
                        new_parent = source_span.parent
                        while not (new_parent.parent is None):
                            new_parents.append(new_parent)
                            if new_parent.parent.span[0] < source_span.span[0]:
                                break
                            new_parent = new_parent.parent
                    if right == len(source_span.subtrees) - 1:
                        # Move up while on right
                        new_parent = source_span.parent
                        while not (new_parent.parent is None):
                            new_parents.append(new_parent)
                            if new_parent.parent.span[1] > source_span.span[1]:
                                break
                            new_parent = new_parent.parent

                for new_parent in new_parents:
                    yield gen_move_successor(source_span, left, right, new_parent, cerrors, gold)

def greedy_search_disco(gold, test):
    # Initialise with the test tree
    cur = (test.clone(), {'type': 'init'}, 0)
    iters = 0
    path = []
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
            if not ntree.check_consistency():
                raise Exception("Inconsistent tree! {}".format(ntree))
            nerrors = parse_errors.get_errors(ntree, gold)
            change = len(cerrors) - len(nerrors)
            if change < 0:
                continue
            if best is None or change > best[2]:
                best = (ntree, info, change)
        cur = best
        iters += 1
    
    for step in path:
        classify(step[1], gold, test)
    
    return (0, iters), path