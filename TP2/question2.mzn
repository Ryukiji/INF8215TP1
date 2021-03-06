% 2162553 Corentin HUBERT 
% 2040063 Adam Mouttaki Bensouda
include "globals.mzn"; 

int: n = 5;
% the given results
array[1..4] of int: res = [1,2,5,7];

% number of points for each player
array[1..n] of var 0..3*n: x;

% (i, j) is the result of player i against player j
array[1..n, 1..n] of var {0,1,3}: matches;

% set x
constraint forall (i in 1..4) ( x[i] = res[i]);


% Diagonale 0
constraint forall (i in 1..n) (matches[i,i] = 0);

% Points distribution
constraint forall (i in 1..n) (
            forall (j in 1..n where i != j) ( 
              if matches[i,j] == 1 then matches[j,i] = 1
              elseif matches[i,j] == 3 then matches[j,i] = 0 
              elseif matches[i,j] == 0 then matches[j,i] = 3 
              else true
              endif 
           ) );


% minimun points by match
constraint forall (i,j in 1..n) ( if i != j then matches[i,j] + matches[j,i] > 0 endif );

% Sum of rows
constraint forall (i in 1..n) (sum(j in 1..n) (matches[i,j]) == x[i]);

solve satisfy;

output 
[
  "Scores\n"
]
++
[
  show(x)
]
++	
[
  "\nMatches\n"
]
++
[ show(matches[i,j]) ++ 
         if j == n then "\n" else " " endif |
         i in 1..n, j in 1..n
]
;