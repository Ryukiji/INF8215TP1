% 2162553 Corentin HUBERT 
% 2040063 Adam Mouttaki Bensouda
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
%number of plantes per row
constraint
  forall(i in 1..n) (
        count([bigGarden[i,j] | j in 1..n], P) == rows[i]
  );

%number of plantes per col
constraint
  forall(i in 1..n) (
        count([bigGarden[j,i] | j in 1..n], P) == cols[i]
  );

% no plant next to other plants
% there can be only one plant in a matrix 3x3 with a that plant being the center, while managing the limits of the big matrix
constraint forall(i, j in 1..n) (
  if bigGarden[i,j] == P
    then count([bigGarden[coordX, coordY] | 
        coordX in if i>1 then i-1 else 1 endif .. if i<n then i+1 else n endif,
        coordY in if j>1 then j-1 else 1 endif .. if j<n then j+1 else n endif], P) == 1
   endif);
% a plant has to be next to a lamp
constraint forall (i,j in 1..n) (
  bigGarden[i,j]==P -> bigGarden[i+1,j]=L \/ bigGarden[i-1,j]=L \/ bigGarden[i,j+1]=L \/ bigGarden[i,j-1]=L 
);
% a lamp has to be next to a plant
constraint forall (i,j in 1..n) (
  bigGarden[i,j]==L -> bigGarden[i+1,j]=P \/ bigGarden[i-1,j]=P \/ bigGarden[i,j+1]=P \/ bigGarden[i,j-1]=P
);
% creating x
constraint forall (i,j in 1..n) (
    x[i,j] == bigGarden[i,j]);
    
solve satisfy;

output 
[
 if j = 1 then "\n" ++ show(rows[i]) ++ " " else " " endif ++
   if fix(x[i,j]) = L then "L" elseif fix(x[i,j]) = P then "P" else "_" endif
| i,j in 1..n
] ++ ["\n  "] ++ [show(cols[i]) ++ " " | i in 1..n];