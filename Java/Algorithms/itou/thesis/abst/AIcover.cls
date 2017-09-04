\NeedsTeXFormat{pLaTeX2e}
\RequirePackage{plext}
\RequirePackage{personal}
\ProvidesClass{AIcover}[2005/12/16 v1.3]
\LoadClass[12pt]{jarticle}
%
%
% 修論, 卒論共通化のための識別子
\newif\ifMorB
%
\DeclareOption{bachelor}{\MorBfalse}	% default
\DeclareOption{master}{\MorBtrue}
%
%
\ExecuteOptions{bachelor}
\ProcessOptions\relax
%
%
%
\setlength{\oddsidemargin}{-8mm}
\setlength{\evensidemargin}{-5.8mm}
\setlength{\topmargin}{0mm}
\setlength{\headheight}{0pc}
\setlength{\headsep}{0pc}
\setlength{\footskip}{2pc}
\setlength{\textheight}{63pc}	%%64pc
%\setlength{\textheight}{250mm}
\setlength{\textwidth}{42pc}
\setlength{\columnsep}{1pc}
\setlength{\columnseprule}{0pt}
%% \oddsidemargin=0mm
%% \topmargin=0mm
\pagestyle{empty}
%
\newcommand{\makeCoverPageI}{%
\vspace*{7mm}
\begin{center}
\setlength{\unitlength}{1mm}
\begin{picture}(134,200)
\put(22, 54){\dashbox{0.6}(90,0){}}
\put(22, 37){\dashbox{0.6}(90,0){}}
\put(22,210){\makebox(90,12){\LARGE%
\ifMorB{修 \ 士 \ 論 \ 文}%
  \else{卒 \ 業 \ 論 \ 文}%
\fi%
}}
\put( 0,189){\makebox(20,5){論文題目}}
\put(22,118){\makebox(20,5){指導教員}}
\put(22, 58){\makebox(20,5){\ifMorB{修了}\else{卒業}\fi{}年度}}
\put(22, 41){\makebox(20,5){学生番号}}
\put(22, 25){\makebox(13,8){氏名}}
\put( 0, 76){\makebox(134,8){\Large
\ifMorB{九州工業大学大学院情報工学府情報科学専攻}
  \else{九州工業大学情報工学部知能情報工学科}
\fi%
}}
\put( 0,143){\dashbox{0.6}(134,43){}}
\put(22, 94){\dashbox{0.6}(90,20){}}
\put(22,  0){\dashbox{0.6}(90,23){}}
%
%
\put(0,143){\makebox(134,43){\large\shortstack{\ThesisTitle}}}
%							% 論文題目
\put(22,94){\makebox(90,20){\large \SupervisorData}}	% 指導教員
\put(22, 0){\makebox(90,23){\large \StudentName}}	% 氏名
\put(22,58){\makebox(90, 5){\large \YearofGrad}}	% 卒業年度
\put(22,41){\makebox(90, 5){\large \StudentIDno}}	% 学生番号
%
%
\end{picture}
\end{center}}
%
%
%
\newcommand{\makeCoverPageII}{%
\begin{center}{\LARGE%
\ifMorB{修 \ 士 \ 論 \ 文}
  \else{卒 \ 業 \ 論 \ 文}
\fi%
}\end{center}
\begin{center}
\parbox{0mm}{\rule{0mm}{40mm}}
\mbox{{\Large\shortstack{\ThesisTitle}}}	% 論文題目
\end{center}
\vspace{15mm}
\begin{center}
\makebox[0mm][r]{\large 指導教員:\hspace{35mm}}
\makebox[0mm]{\Large \SupervisorData}		% 指導教員
\end{center}
\vspace{30mm}
\begin{center}{\Large
\ifMorB{九州工業大学大学院 \\ 情報工学府 情報科学専攻}
  \else{九州工業大学情報工学部 \\ 知能情報工学科}
\fi%
}\end{center}
\vspace{15mm}
\begin{center}
{\large \YearofGrad} 年度
\end{center}
\vspace{15mm}
\begin{center}
{\Large \StudentName}				% 学生氏名
\end{center}
}
%
%
%
\newcommand{\makeSpine}{%
\begin{flushleft}
\setlength{\unitlength}{1mm}
\begin{picture}(13,246)
\thicklines
\put(0,  0){\line(1,0){13}}
\put(0, 56){\line(1,0){13}}
\put(0, 64){\line(1,0){13}}
\put(0, 80){\line(1,0){13}}
\put(0, 88){\line(1,0){13}}
\put(0,238){\line(1,0){13}}
\put(0,246){\line(1,0){13}}
\put(0,  0){\line(0,1){246}}
\put(13, 0){\line(0,1){246}}
\put(0,238){\makebox(13,8){題目}}
\put(0, 80){\makebox(13,8){年度}}
\put(0, 56){\makebox(13,8){氏名}}
%
% 題目の変更
%
\put(0, 88){\makebox(13,150)
%%%%% 1行用
%% {\Large \parbox<t>{20zw}{\\input{title}}}}
%%%%% 2行用
{\large \shortstack{\parbox<t>{30zw}{\ThesisTitle}}}}
%
%
%
\put(0, 64){\makebox(13,16){\Large \YearofGrad}}
\put(0,  0){\makebox(13,56){\Large\shortstack{\parbox<t>{7zw}{\StudentName}}}}
\end{picture}
\end{flushleft}
}

%% end of AIcover.cls
