\NeedsTeXFormat{pLaTeX2e}
\RequirePackage{personal}
\ProvidesClass{AIabst}[2006/12/21 v1.3]
\LoadClass[10pt]{jarticle}
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
%
% レイアウト
\setlength{\oddsidemargin}{-5mm}
\setlength{\evensidemargin}{-5.8mm}
\setlength{\topmargin}{-9mm}
\setlength{\headheight}{0pc}
\setlength{\headsep}{0pc}
\setlength{\footskip}{2pc}
%\setlength{\textheight}{66pc}
\setlength{\textheight}{260mm}
%\setlength{\textwidth}{40pc}
\setlength{\textwidth}{170mm}
\setlength{\columnsep}{2pc}
\setlength{\columnseprule}{0pt}

% ページ番号
\pagestyle{empty}

% セクションマクロの変更
\renewcommand{\section}{\@startsection {section}{1}{\z@}%
   {1ex \@plus .5ex \@minus .2ex}%
   {.5ex \@plus.3ex}%
   {\reset@font\large\bfseries}}
%%    {3.5ex \@plus 1ex \@minus .2ex}%
%%    {2.3ex \@plus.2ex}%
%%    {\reset@font\sec@font}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
   {.5ex\@plus .5ex \@minus .2ex}%
   {.2ex \@plus .3ex}%
   {\reset@font\normalsize\bfseries}}
%%    {3.25ex\@plus 1ex \@minus .2ex}%
%%    {1.5ex \@plus .2ex}%
%%    {\reset@font\ssec@font}}

% マクロ
\def\p{\\[-1mm]}

\newcommand{\makeAbstHeader}{%
\twocolumn[
\begin{center}
\ifMorB{\Large 論　文　概　要}
  \else{九州工業大学 情報工学部 知能情報工学科 \YearofGrad 年度 卒業論文 概要}
\fi%
\end{center}
\begin{center}
\setlength{\unitlength}{1mm}
\ifMorB{%
九州工業大学大学院情報工学府\quad
情報科学専攻\quad
知能情報工学分野\quad
\\[1.5mm]
\begin{picture}(160,21)
\thicklines
\put(  0, 0){\framebox(160,21){}}
\put(  0,14){\line(1,0){160}}
%\put(  0,21){\line(1,0){160}}
\put( 27, 0){\line(0,1){21}}
%\put( 54,21){\line(0,1){7}}
\put( 80,14){\line(0,1){7}}
\put(107,14){\line(0,1){7}}
%\put(  0,21){\makebox(54,7){所属の専門分野及び講座名}}
%\put( 54,21){\makebox(54,7){\large 知能情報工学}}
%\put(100,21){\makebox(53,7){\large \DivisionName 講座}}
%% \put(120,21){\makebox(53,7){\large 講座}}
\put(  0,14){\makebox(27,7){学 生 番 号}}
\put( 27,14){\makebox(53,7){\large \StudentIDno}}
\put( 80,14){\makebox(27,7){氏\qquad 名}}
\put(107,14){\makebox(53,7){\large \StudentName}}
\put(  0, 0){\makebox( 27,14){論 文 題 目}}
\put( 27, 0){\makebox(133,14){\large\shortstack{\ThesisTitle}}}
\end{picture}
}%
\else{%
\begin{picture}(160,28)
\thicklines
\put(  0, 0){\framebox(160,28){}}
\put(  0,14){\line(1,0){160}}
\put(  0,21){\line(1,0){160}}
\put( 27, 0){\line(0,1){28}}
\put( 80,14){\line(0,1){14}}
\put(107,14){\line(0,1){14}}
%%
\put(  0,21){\makebox(27,7){所 属 講 座}}
\put( 25,21){\makebox(54,7){\large \DivisionName}}
\put(80,21){\makebox(27,7){指 導 教 員}}
\put(105,21){\makebox(53,7){\large \SupervisorData}}
\put(  0,14){\makebox(27,7){学 生 番 号}}
\put( 27,14){\makebox(53,7){\large \StudentIDno}}
\put( 80,14){\makebox(27,7){氏\qquad 名}}
\put(107,14){\makebox(53,7){\large \StudentName}}
\put(  0, 0){\makebox( 27,14){論 文 題 目}}
\put( 27, 0){\makebox(133,14){\large\shortstack{\ThesisTitle}}}
\end{picture}
}%
\fi%
\vspace{8pt}
\end{center}
]
}

%% end of AIabst.cls
