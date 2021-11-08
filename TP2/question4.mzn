% 2162553 Corentin HUBERT 
% 2040063 Adam Mouttaki Bensouda

% Solver COIN-BC 2.10.5/1.17.5 works best

include "globals.mzn";

int: n_nurse; % Number of nurses
int: n_week; % Number of weeks in our horizon
int: n_day; % Number of days in our horizon (7 * n_week). Monday is the first day of the week
int: n_work_period; % Number of work period. It equals 4 (MATIN, SOIREE, NUIT, REPOS)

% Indexing the work periods
int: MATIN = 1;
int: SOIREE = 2;
int: NUIT = 3;
int: REPOS = 4;

int: dur_work_period; % Duration of the work periods (6h)

array[1..n_day,1..n_work_period] of int: demande_min; % Number of nurses required for each day and each work period (= 0 for REPOS)
array[1..n_nurse,1..n_day,1..n_work_period] of int: preference; % Work preference for each nurse

% Decision variables
array[1..n_nurse,1..n_day,1..n_work_period] of var 0..1: x; % work period [MATIN, SOIREE, NUIT, REPOS]
var int: sum_preference;


% On définit la somme des préférences
constraint sum(i in 1..n_nurse, j in 1..n_day, k in 1..4)(preference[i,j,k]*x[i,j,k]) = sum_preference;

% On remplit entièrement l'emploi du temps de chaque infirmière
constraint forall (i in 1..n_nurse) (
  forall (j in 1..n_day) (
    count([x[i,j,k] | k in 1..4], 1) = 1
  )
);

% Demande journalière minimale
constraint forall (j in 1..n_day) (
  count([x[i,j,MATIN] | i in 1..n_nurse], 1) >= demande_min[j,MATIN]
);

constraint forall (j in 1..n_day) (
  count([x[i,j,SOIREE] | i in 1..n_nurse], 1) >= demande_min[j,SOIREE]
);

constraint forall (j in 1..n_day) (
  count([x[i,j,NUIT] | i in 1..n_nurse], 1) >= demande_min[j,NUIT]
);

% Une infirmière fait au maximum un créneau par jour
constraint forall (j in 1..n_day) (
  forall (i in 1..n_nurse) (  
    count([x[i,j,k] | k in 1..3], 1) <= 1
  )
);

% 36h max par semaine <-> au moins un jour de repos par semaine
constraint forall (n_w in 0..n_week-1) (
  forall (i in 1..n_nurse) (
    count([x[i, n_w*7 + n_d,REPOS] | n_d in 1..7], 1) >= 1
  )
);

% Une infirmière doit travailler au moins 4 fois par semaine <-> maximum 3 jour de repos par semaine
constraint forall (n_w in 0..n_week-1) (
  forall (i in 1..n_nurse) (
    count([x[i, n_w*7 + n_d,4] | n_d in 1..7], 1) <= 3
  )
);

% Un travail de nuit est suivi d'un jour de repos
constraint forall (j in 1..n_day-1) (
  forall (i in 1..n_nurse) (
    if x[i,j,NUIT] == 1 then x[i,j+1,REPOS]=1 endif
  )
);

% Impossible d'avoir un jour de travail isolé
constraint forall (j in 2..n_day-1) (
  forall (i in 1..n_nurse) (
    (x[i,j,NUIT] == 1 \/ x[i,j,MATIN] == 1 \/ x[i,j,SOIREE] == 1) -> (x[i,j+1,REPOS]!=1 \/ x[i,j-1,REPOS]!=1)
  )
);

% Impossible d'avoir plus de 3 jours de travail consécutif
constraint forall (j in 3..n_day-2) (
  forall (i in 1..n_nurse) (
    (x[i,j,REPOS] != 1 /\ x[i,j+1,REPOS] != 1 /\ x[i,j-1,REPOS] != 1) -> (x[i,j+2,REPOS]=1 /\ x[i,j-2,REPOS]==1)
  )
);

solve maximize sum_preference;

output
[
  "total: " ++ show(sum_preference) ++ "\n" ++
  "solution:\n\t    "
]
++
["Day \(j)\t" | j in 1..n_day]
++ 
[ 
  if j = 1 then "\n NURSE \(i): " else "\t" endif ++
    if fix(x[i,j,4]) == 1 then "--" elseif fix(x[i,j,3]) == 1  then "NUIT" elseif fix(x[i,j,2]) == 1  then "SOIREE" elseif fix(x[i,j,1]) == 1 then "MATIN" else "NONE" endif
  | i in 1..n_nurse, j in 1..n_day
];

