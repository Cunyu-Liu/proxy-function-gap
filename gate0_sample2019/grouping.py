def find(parent, item):
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union(parent, sizes, left, right):
    left_root = find(parent, left)
    right_root = find(parent, right)
    if left_root == right_root:
        return
    if sizes[left_root] < sizes[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    sizes[left_root] += sizes[right_root]


def partition_violations(edges, assignments):
    violations = []
    for left, right in edges:
        if assignments.get(left) != assignments.get(right):
            violations.append({"left": left, "right": right, "left_group": assignments.get(left), "right_group": assignments.get(right)})
    return violations


def exact_assignment_violations(records):
    seen = {}
    violations = []
    for record in records:
        sequence_id = record["sequence_id"]
        group_id = record["near_group_id"]
        if sequence_id in seen and seen[sequence_id][0] != group_id:
            violations.append({
                "sequence_id": sequence_id,
                "first_raw_row_id": seen[sequence_id][1],
                "first_group_id": seen[sequence_id][0],
                "raw_row_id": record["raw_row_id"],
                "group_id": group_id,
            })
        else:
            seen[sequence_id] = (group_id, record["raw_row_id"])
    return violations
