include "globals.mzn";

par int: nr_alignments;
par int: input_length;

par int: max_id = input_length;       %TODO: can we find a better upper bound of the cluster_id?

array[1..input_length] of var 1..max_id: input;
array[1..nr_alignments+1] of var 1..input_length+1: subalignment_start;

function var int: subalignment_length(int: a) = subalignment_start[a+1]-subalignment_start[a];

% This is the output.
array[1..input_length] of var 1..max_id: cluster_ids;

% Cluster ids should be ordered such that for all alignments, the higher columns are assigned a higher cluster id. 
% This is the actual ordering asked by the problem definition.
constraint forall(a in 1..nr_alignments where subalignment_length(a) > 1, 
                  n1 in subalignment_start[a]..subalignment_start[a+1]-2) 
      (cluster_ids[n1] < cluster_ids[n1+1]);

%% Nodes that were not in the same cluster in the input, cannot be in the same cluster in the output.
%This means that clusters can only be broken in the solution.
constraint forall (n1 in 1..input_length-1,
                   n2 in n1+1..input_length)
        (if input[n1] != input[n2]
        then cluster_ids[n1] != cluster_ids[n2]
        endif);

constraint cluster_ids[1] = 1;

% Solve for the least possible cluster id's assigned.
solve :: int_search(cluster_ids, input_order, indomain_min)
        minimize max(cluster_ids); % TODO: Check if it would be more efficient to only check for the max value of the last values for each subalignment.

% output["max_id = " ++ show(max(cluster_ids)) ++ "\n" ++ 
%        join("\n", [ join(", ", [show(cluster_ids[a, c]) | c in 1..max_cols]) | a in 1..nr_alignments])]  ;

output[ join(",", [show(cluster_ids[i]) | i in 1..input_length])];

% ------------- TODO -------------
%
% 1. How to test search heuristics without overtraining it on the testdata that we received?
% -> Probably need to find extra datasets.
% 