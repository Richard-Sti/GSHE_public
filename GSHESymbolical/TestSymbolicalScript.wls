#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title:: *)
(*Gravitational spin Hall effect*)


(* ::Subtitle:: *)
(*Coordinates*)




(* Assume all variables are real *)
$Assumptions=Element[x0[\[Tau]],Reals]&&Element[x1[\[Tau]],Reals]&&Element[x2[\[Tau]],Reals]&&Element[x3[\[Tau]],Reals]&&Element[p0[\[Tau]],Reals]&&Element[p1[\[Tau]],Reals]&&Element[p2[\[Tau]],Reals]&&Element[p3[\[Tau]],Reals]&&Element[s,Reals]&&Element[v,Reals]&&Element[ro,Reals]&&Element[\[Theta]o,Reals]&&x1[\[Tau]]>2M&&Element[a,Reals]&&0<x2[\[Tau]]<\[Pi];
(* Coordinates *)

(* X^\[Mu] *)
X={x0[\[Tau]],x1[\[Tau]],x2[\[Tau]],x3[\[Tau]]};

(* Subscript[P, \[Mu]] *)
P={p0[\[Tau]],p1[\[Tau]],p2[\[Tau]],p3[\[Tau]]};

(* Subscript[P, i] *)
p={p1[\[Tau]],p2[\[Tau]],p3[\[Tau]]};


(* BL coordinates in Kerr (t, r, \[Theta], \[Phi]) = (x0, x1, x2, x3) *)

M=1;  (* mass *)
rs = 2M; (* Schwarzschild radius *)

\[CapitalSigma]=r^2+a^2 Cos[\[Theta]]^2/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
\[CapitalDelta]=r^2-rs r+a^2/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
A=(r^2+a^2)^2-a^2 \[CapitalDelta] Sin[\[Theta]]^2/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];

(* Metric Subscript[g, \[Mu]\[Nu]] *)

g=\!\(\*
TagBox[
RowBox[{"(", "", GridBox[{
{
RowBox[{"-", 
RowBox[{"(", 
RowBox[{"1", "-", 
FractionBox[
RowBox[{"rs", " ", "r"}], "\[CapitalSigma]"]}], ")"}]}], "0", "0", 
RowBox[{"-", 
FractionBox[
RowBox[{" ", 
RowBox[{"rs", " ", "a", " ", "r", " ", 
SuperscriptBox[
RowBox[{"Sin", "[", "\[Theta]", "]"}], "2"]}]}], "\[CapitalSigma]"]}]},
{"0", 
FractionBox["\[CapitalSigma]", "\[CapitalDelta]"], "0", "0"},
{"0", "0", "\[CapitalSigma]", "0"},
{
RowBox[{"-", 
FractionBox[
RowBox[{"rs", " ", "a", " ", "r", " ", 
SuperscriptBox[
RowBox[{"Sin", "[", "\[Theta]", "]"}], "2"]}], "\[CapitalSigma]"]}], "0", "0", 
RowBox[{"A", 
FractionBox[
SuperscriptBox[
RowBox[{"Sin", "[", "\[Theta]", "]"}], "2"], "\[CapitalSigma]"]}]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}}], "", ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\)/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
(* Inverse metric g^\[Mu]\[Nu] *)

ig=Inverse[g]//Simplify;

(* Orthonormal tetrad (Subscript[e, a])^\[Mu] and dual tetrad Subscript[(d^a), \[Mu]], with (Subscript[e, a])^\[Mu] = t^\[Mu] *)

f0={(r^2+a^2)/Sqrt[\[CapitalSigma] \[CapitalDelta]],0,0,a/Sqrt[\[CapitalSigma] \[CapitalDelta]]}/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
f1={0,Sqrt[\[CapitalDelta]/\[CapitalSigma]],0,0}/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
f2={0,0,Sqrt[1/\[CapitalSigma]],0}/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];
f3={(a Sin[\[Theta]]^2)/(Sqrt[\[CapitalSigma]]Sin[\[Theta]]),0,0,1/(Sqrt[\[CapitalSigma]]Sin[\[Theta]])}/.r->x1[\[Tau]]/.\[Theta]->x2[\[Tau]]/.\[Phi]->x3[\[Tau]];


e0=(f0+v[x1[\[Tau]]] f3)/Sqrt[1-v[x1[\[Tau]]]^2];
e1=f1;
e2=f2;
e3=(f3+v[x1[\[Tau]]] f0)/Sqrt[1-v[x1[\[Tau]]]^2];
d0=(e0 . g)//Simplify;
d1=(e1 . g)//Simplify;
d2=(e2 . g)//Simplify;
d3=(e3 . g)//Simplify; 
T=e0;
Td=d0;

Detg=Det[g]//Simplify;

(* Christoffel Subscript[\[CapitalGamma]^i, jk] = 1/2g^im( \!\(
\*SubscriptBox[\(\[PartialD]\), \(k\)]
\*SubscriptBox[\(g\), \(mj\)]\) + \!\(
\*SubscriptBox[\(\[PartialD]\), \(j\)]
\*SubscriptBox[\(g\), \(mk\)]\) - \!\(
\*SubscriptBox[\(\[PartialD]\), \(m\)]
\*SubscriptBox[\(g\), \(jk\)]\) ) *)

\[CapitalGamma]=1/2 Parallelize[Table[(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(m = 1\), \(4\)]\((\(ig[\([i]\)]\)[\([m]\)] \((D[\(g[\([m]\)]\)[\([j]\)], X[\([k]\)]] + D[\(g[\([m]\)]\)[\([k]\)], X[\([j]\)]] - D[\(g[\([j]\)]\)[\([k]\)], X[\([m]\)]])\))\)\))//Simplify,{i,1,4},{j,1,4},{k,1,4}]];

(* Riemann Subscript[R^i, jkl] = \!\(
\*SubscriptBox[\(\[PartialD]\), \(k\)]
\*SubscriptBox[
SuperscriptBox[\(\[CapitalGamma]\), \(i\)], \(lj\)]\) - \!\(
\*SubscriptBox[\(\[PartialD]\), \(l\)]
\*SubscriptBox[
SuperscriptBox[\(\[CapitalGamma]\), \(i\)], \(kj\)]\) + Subscript[\[CapitalGamma]^i, km]Subscript[\[CapitalGamma]^m, lj] - Subscript[\[CapitalGamma]^i, lm]Subscript[\[CapitalGamma]^m, kj] *)

Riem=Table[(D[\[CapitalGamma][[i]][[l]][[j]],X[[k]]]-D[\[CapitalGamma][[i]][[k]][[j]],X[[l]]]+\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(m = 1\), \(4\)]\((\(\(\[CapitalGamma][\([i]\)]\)[\([k]\)]\)[\([m]\)] \(\(\[CapitalGamma][\([m]\)]\)[\([l]\)]\)[\([j]\)] - \(\(\[CapitalGamma][\([i]\)]\)[\([l]\)]\)[\([m]\)] \(\(\[CapitalGamma][\([m]\)]\)[\([k]\)]\)[\([j]\)])\)\))//Simplify,{i,1,4},{j,1,4},{k,1,4},{l,1,4}]//Parallelize;

(* Riemann Subscript[R, ijkl] *)
R=Table[\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i1 = 1\), \(4\)]\(\(g[\([i1]\)]\)[\([i]\)] \(\(\(Riem[\([i1]\)]\)[\([j]\)]\)[\([k]\)]\)[\([l]\)]\)\)//Simplify,{i,1,4},{j,1,4},{k,1,4},{l,1,4}]//Parallelize;

(* P^\[Mu] = g^\[Mu]\[Alpha]Subscript[P, \[Alpha]] *)

Pu=Table[(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Alpha] = 1\), \(4\)]\((\(ig[\([\[Mu]]\)]\)[\([\[Alpha]]\)] P[\([\[Alpha]]\)])\)\))//Simplify,{\[Mu],1,4}]//Parallelize;

(* H = 1/2g^\[Mu]\[Nu]Subscript[P, \[Mu]]Subscript[P, \[Nu]] = 0  =>  Subscript[P, 0] = 1/g^00(-g^(0i)Subscript[P, i] + Sqrt[(g^(0i)Subscript[P, i])^2 - g^00g^ijSubscript[P, i]Subscript[P, j]]) *)
pt0=ig[[1]][[1]]^-1 (-(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i = 2\), \(4\)]\((\(ig[\([1]\)]\)[\([i]\)] P[\([i]\)])\)\))+((\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i = 2\), \(4\)]\((\(ig[\([\)\(1\)\(]\)]\)[\([\)\(i\)\(]\)] P[\([\)\(i\)\(]\)])\)\))^2-ig[[1]][[1]](\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i = 2\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(j = 2\), \(4\)]\((\(ig[\([\)\(i\)\(]\)]\)[\([\)\(j\)\(]\)] P[\([\)\(i\)\(]\)] P[\([\)\(j\)\(]\)])\)\)\)))^(1/2))//Simplify;


(* Spin tensor Subscript[S, \[Alpha]\[Beta]] = -(s/(T^\[Mu]Subscript[P, \[Mu]]))Subscript[\[Epsilon], \[Alpha]\[Beta]\[Gamma]\[Delta]]P^\[Gamma]T^\[Delta] *)

S=ParallelTable[FullSimplify[-s/T . P Sqrt[-Detg]*\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Gamma] = 1\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Delta] = 1\), \(4\)]\ \(\(\(\(LeviCivitaTensor[4]\)[\([\[Alpha]]\)]\)[\([\[Beta]]\)]\)[\([\[Gamma]]\)]\)[\([\[Delta]]\)] Pu[\([\[Gamma]]\)] T[\([\[Delta]]\)]\)\)],{\[Alpha],1,4},{\[Beta],1,4}]; 
Su=ParallelTable[Simplify[\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Gamma] = 1\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Delta] = 1\), \(4\)]\(S[\([\[Gamma]]\)]\)[\([\[Delta]]\)] \(ig[\([\[Gamma]]\)]\)[\([\[Alpha]]\)] \(ig[\([\[Delta]]\)]\)[\([\[Beta]]\)]\)\)],{\[Alpha],1,4},{\[Beta],1,4}];

(* \!\(
\*SubscriptBox[\(\[Del]\), \(\[Mu]\)]
\*SubscriptBox[\(T\), \(\[Alpha]\)]\) = \!\(
\*SubscriptBox[\(\[PartialD]\), \(\[Mu]\)]
\*SubscriptBox[\(T\), \(\[Alpha]\)]\) - Subscript[\[CapitalGamma]^\[Rho], \[Mu]\[Alpha]]Subscript[T, \[Rho]] *)
DTd=ParallelTable[(D[Td[[\[Alpha]]],X[[\[Mu]]]]-\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Rho] = 1\), \(4\)]\((\(\(\[CapitalGamma][\([\[Rho]]\)]\)[\([\[Mu]]\)]\)[\([\[Alpha]]\)] Td[\([\[Rho]]\)])\)\))//Simplify,{\[Mu],1,4},{\[Alpha],1,4}];
(* P^\[Mu]\!\(
\*SubscriptBox[\(\[Del]\), \(\[Mu]\)]
\*SubscriptBox[\(T\), \(\[Alpha]\)]\) = P^\[Mu](\!\(
\*SubscriptBox[\(\[PartialD]\), \(\[Mu]\)]
\*SubscriptBox[\(T\), \(\[Alpha]\)]\) - Subscript[\[CapitalGamma]^\[Rho], \[Mu]\[Alpha]]Subscript[T, \[Rho]]) *)
PDTd=ParallelTable[(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Mu] = 1\), \(4\)]\(Pu[\([\[Mu]]\)] \(DTd[\([\[Mu]]\)]\)[\([\[Alpha]]\)]\)\))//Simplify,{\[Alpha],1,4}];
(* \[Epsilon]/(T^\[Rho]Subscript[P, \[Rho]])S^\[Mu]\[Alpha]P^\[Nu]\!\(
\*SubscriptBox[\(\[Del]\), \(\[Nu]\)]
\*SubscriptBox[\(T\), \(\[Alpha]\)]\) *)
dx1=ParallelTable[\[Epsilon]/T . P (\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Alpha] = 1\), \(4\)]\(\(Su[\([\[Mu]]\)]\)[\([\[Alpha]]\)] PDTd[\([\[Alpha]]\)]\)\)),{\[Mu],1,4}];
(* Subscript[\[CapitalGamma]^\[Alpha], \[Beta]\[Mu]]Subscript[P, \[Alpha]]Overscript[x, .]^\[Beta] *)

Pd=Table[(\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Alpha] = 1\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(\[Beta] = 1\), \(4\)]\((\(\(\[CapitalGamma][\([\[Alpha]]\)]\)[\([\[Beta]]\)]\)[\([\[Mu]]\)] P[\([\[Alpha]]\)] \((Pu[\([\[Beta]]\)] + dx1[\([\[Beta]]\)])\))\)\)\)),{\[Mu],1,4}]//Parallelize;
RS=Parallelize[Table[\[Epsilon]/2 \!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i = 1\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(j = 1\), \(4\)]\(
\*UnderoverscriptBox[\(\[Sum]\), \(k = 1\), \(4\)]Pu[\([i]\)] \(Su[\([j]\)]\)[\([k]\)] \(\(\(R[\([j]\)]\)[\([k]\)]\)[\([\[Alpha]]\)]\)[\([i]\)]\)\)\)//Simplify,{\[Alpha],1,4}]];
pd=Table[Pd[[i]],{i,2,4}];
Rs=Table[RS[[i]],{i,2,4}];

Xdot=Parallelize[Table[(Pu[[i]]+dx1[[i]])/.p0[\[Tau]]->pt0,{i,1,4}]];
pdot=Parallelize[Table[(pd[[i]]-Rs[[i]])/.p0[\[Tau]]->pt0,{i,1,3}]];
\!\(\*
TagBox[
RowBox[{"(", "", GridBox[{
{
RowBox[{
RowBox[{"-", "1"}], "+", 
FractionBox[
RowBox[{"2", " ", 
RowBox[{"x1", "[", "\[Tau]", "]"}]}], 
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}]]}], "0", "0", 
RowBox[{"-", 
FractionBox[
RowBox[{"2", " ", "a", " ", 
SuperscriptBox[
RowBox[{"Sin", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"], " ", 
RowBox[{"x1", "[", "\[Tau]", "]"}]}], 
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}]]}]},
{"0", 
FractionBox[
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}], 
RowBox[{
SuperscriptBox["a", "2"], "-", 
RowBox[{"2", " ", 
RowBox[{"x1", "[", "\[Tau]", "]"}]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}]], "0", "0"},
{"0", "0", 
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}], "0"},
{
RowBox[{"-", 
FractionBox[
RowBox[{"2", " ", "a", " ", 
SuperscriptBox[
RowBox[{"Sin", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"], " ", 
RowBox[{"x1", "[", "\[Tau]", "]"}]}], 
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}]]}], "0", "0", 
FractionBox[
RowBox[{
SuperscriptBox[
RowBox[{"Sin", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"], " ", 
RowBox[{"(", 
RowBox[{
SuperscriptBox[
RowBox[{"(", 
RowBox[{
SuperscriptBox["a", "2"], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}], ")"}], "2"], "-", 
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Sin", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"], " ", 
RowBox[{"(", 
RowBox[{
SuperscriptBox["a", "2"], "-", 
RowBox[{"2", " ", 
RowBox[{"x1", "[", "\[Tau]", "]"}]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}], ")"}]}]}], ")"}]}], 
RowBox[{
RowBox[{
SuperscriptBox["a", "2"], " ", 
SuperscriptBox[
RowBox[{"Cos", "[", 
RowBox[{"x2", "[", "\[Tau]", "]"}], "]"}], "2"]}], "+", 
SuperscriptBox[
RowBox[{"x1", "[", "\[Tau]", "]"}], "2"]}]]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "Rows" -> {{Baseline}}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}}], "", ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\)
