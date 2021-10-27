include "globals.mzn";

int: n; % num rows and columns
int: L = 1; % Lamp
int: P = 2; % Plant
% 1 for a lamp, 2 for a plant and 0 for nothing 
array[1..n,1..n] of 0..2: garden;
% number of plants by row/col
array[1..n] of int: rows;
array[1..n] of int: cols;
% decision variables
array[1..n,1..n] of var 0..P: x;
% garden with two colums and two rows more
array[0..n+1,0..n+1] of var 0..P: bigGarden;
% initialing big garden
constraint forall (i,j in 1..n) (
    if garden[i,j]>0 then bigGarden[i,j] = garden[i,j] else true endif
);
constraint forall (i in 0..n+1) (bigGarden[0,i] == 0);
constraint forall (i in 0..n+1) (bigGarden[i,0] == 0);
constraint forall (i in 0..n+1) (bigGarden[n+1,i] == 0);
constraint forall (i in 0..n+1) (bigGarden[i,n+1] == 0);
% number of plants on rows
constraint forall (i in 1..n) (sum(j in 1..n) ((bigGarden[i,j] - garden[i,j]) div 2) == rows[i]);
% number of plants on cols
constraint forall (i in 1..n) (sum(j in 1..n) ((bigGarden[j,i] - garden[j,i]) div 2) == cols[i]);

% a plant has to be next to a lamp
% constraint forall (i,j in 1..n) (
%   bigGarden[i,j]==P -> bigGarden[i+1,j]=L \/ bigGarden[i-1,j]=L \/ bigGarden[i,j+1]=L \/ bigGarden[i,j-1]=L 
% );
% % a lamp has to be next to a plant
% constraint forall (i,j in 1..n) (
%   bigGarden[i,j]==L -> bigGarden[i+1,j]=P \/ bigGarden[i-1,j]=P \/ bigGarden[i,j+1]=P \/ bigGarden[i,j-1]=P
% );
% creating x
constraint forall (i,j in 1..n) (
    x[i,j] == bigGarden[i,j]);
    
    
    
% no plant next to other plants
constraint forall (i,j in 1..n) (
  bigGarden[i,j]==P -> bigGarden[i+1,j]!=P /\ bigGarden[i+1,j+1]!=P /\ bigGarden[i, j+1]!=P /\ bigGarden[i-1, j+1]!=P /\ bigGarden[i+1, j-1]!=P /\ bigGarden[i, j-1]!=P /\ bigGarden[i-1, j-1]!=P
);

% constraint forall (i,j in 1..n) (
%   bigGarden[i,j]==P -> bigGarden[i+1,j+1]!=P /\ bigGarden[i, j+1]!=P /\ bigGarden[i-1, j+1]!=P /\ bigGarden[i+1, j-1]!=P /\ bigGarden[i, j-1]!=P /\ bigGarden[i-1, j-1]!=P
% );


solve satisfy;
output 
[
 if j = 1 then "\n" ++ show(rows[i]) ++ " " else " " endif ++
   if fix(x[i,j]) = L then "L" elseif fix(x[i,j]) = P then "P" else "_" endif
| i,j in 1..n
] ++ ["\n  "] ++ [show(cols[i]) ++ " " | i in 1..n];