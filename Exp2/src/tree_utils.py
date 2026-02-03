from sklearn.tree import export_text

def print_tree_structure(clf, feature_names):
    print("\nDecision Tree Structure:")
    tree_rules = export_text(clf, feature_names=feature_names)
    print(tree_rules)

def print_gini_indices(clf, feature_names):
    print("\nNode Gini Indices:")
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    impurity = clf.tree_.impurity

    node_depth = [0] * n_nodes
    is_leaves = [False] * n_nodes
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        if is_leaves[i]:
            print(f"Node {i} (Leaf): Gini = {impurity[i]:.4f}")
        else:
            print(f"Node {i} (Split on {feature_names[feature[i]]} <= {threshold[i]:.2f}): Gini = {impurity[i]:.4f}")

def print_feature_importance(clf, feature_names):
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, clf.feature_importances_):
        print(f"{name}: {importance:.4f}")
