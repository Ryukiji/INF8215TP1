include "globals.mzn";


int: r; % number of rows 
int: c; % number of columns
array[1..r] of int: rowinsects; % total of insects by row
array[1..c] of int: colinsects; % total of insects by col
array[1..r, 1..c] of 0..r*c: knownpositions;  % known number of insects, 0 appears when unknown
array[1..r, 1..c] of var 1..r*c: x; % the final garden grid (decision variable)
%fill initial garden
constraint forall(i in 1..r)(
           forall(j in 1..c)(
                 if knownpositions[i,j] > 0 then x[i,j] = knownpositions[i,j] else true endif ));
    
% All different in rows 
constraint forall (i in 1..r) (
                   alldifferent( [ x[i,j] | j in 1..c ]) ); 
% All different in columns
constraint forall (j in 1..c) (
                   alldifferent( [ x[i,j] | i in 1..r ]) ); 
                   
% Sum of rows
constraint forall (i in 1..r) (sum(j in 1..c) (x[i,j]) == rowinsects[i]);
% Sum of columns
constraint forall (j in 1..c) (sum(i in 1..r) (x[i,j]) == colinsects[j]);
solve satisfy;
output [
  "["
]
++
[
  if j = 1 then "\n" else "," endif ++
    if fix(x[i,j]) < 10 then "  " else " "  endif ++
    show(x[i,j]) ++
  if j = c then 
    ", f(" ++ show(rowinsects[i]) ++ "),"
  else 
    ""
  endif
  | i in 1..r, j in 1..c
] 
++
["\n"]++
[
  "f(" ++ show(colinsects[j]) ++ "),"
  | j in 1..c
]
++ ["]\n"]
;