include "globals.mzn";

par int: nr_alignments;
par int: max_cols;

par int: max_id = nr_alignments*max_cols;       %TODO: can we find a better upper bound of the cluster_id?
                                                % Will  be done in the python code.

array[1..nr_alignments, 1..max_cols] of var 0..max_id: input;

% This is the output.
array[1..nr_alignments, 1..max_cols] of var 0..max_id: cluster_ids;

% A rasterscan order is assigned to the nodes by defining all cells in the matrix (so each node) with a unique integer.
% The node in the left upper corner receives the number 0 as identifier. The second cell receives id 1. 
% The first cell of the second row receives id (max_cols*1). The second cell of the second row receives id (max_cols*1) +1 and so on.
function int: rasterscan_order(int: row, int: col) = (row-1)*max_cols+col; % From zero to rows*cols

% Cluster ids should be ordered such that for all alignments, the higher columns are assigned a higher cluster id. 
% This is the actual ordering asked by the problem definition.
%% constraint forall(a in 1..nr_alignments, c in 1..max_cols-1)
%%   (if cluster_ids[a,c+1] != 0
%%   then cluster_ids[a, c] < cluster_ids[a, c+1]
%%   endif);

constraint forall(a in 1..nr_alignments, c1 in 1..max_cols-1, c2 in c1+1..max_cols)
  (if cluster_ids[a,c1] != 0 /\ cluster_ids[a,c2] != 0
  then cluster_ids[a, c1] < cluster_ids[a, c2]
  endif);

%% Nodes that were not in the same cluster in the input, cannot be in the same cluster in the output.
constraint forall (a1 in 1..nr_alignments,
                   c1 in 1..max_cols,
                   a2 in 1..nr_alignments,
                   c2 in 1..max_cols)
        (if rasterscan_order(a1, c1) < rasterscan_order(a2, c2) % Symmetry breaking
         /\ input[a1, c1] != input[a2, c2]
         then cluster_ids[a1,c1] != cluster_ids[a2,c2]
         else true endif);

% Nodes marked with cluster id 0 in the input are non existing nodes. They should have cluster id 0 as well.
% Nodes marked with cluster id n > 0 in the input should get a cluster id.
constraint forall (a in 1..nr_alignments, c in 1..max_cols)
        (if input[a,c] == 0
        then cluster_ids[a, c] == 0
        else cluster_ids[a,c] != 0 %TODO: maybe > 0 is more efficient?
        endif);

%constraint cluster_ids[1,1] = 1;

% Solve for the least possible cluster id's assigned.
solve minimize max(cluster_ids); % TODO: Check if it would be more efficient to only check for the max value of last non-zero values.

output["max_id = " ++ show(max(cluster_ids)) ++ "\n" ++ 
       join("\n", [ join(", ", [show(cluster_ids[a, c]) | c in 1..max_cols]) | a in 1..nr_alignments])]  ;

% ------------- TODO -------------
%
% 1. How to test search heuristics without overtraining it on the testdata that we received?
% -> Probably need to find extra datasets.
% 