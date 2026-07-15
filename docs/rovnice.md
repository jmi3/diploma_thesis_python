# Vlnová rovnice

Máme vlnovou rovnici pro skalární pole $\psi$ jako
$$
(\square - F)\psi = 0,
$$

kterou chápeme v souřadnicích (sférická symetrie) $(T,\ R)$.

Po přepisu do prvního řádu dostáváme systém rovnic pro proměnné $\psi$, $\phi_R \equiv \partial_R\psi$ a $\pi \equiv -\partial_T\psi$ jako

$$
\begin{align*}
    \partial_T \psi &= -\pi, \\[0.3em]
    \partial_T \phi_R &= -\partial_R \pi + \gamma_2 (\partial_R\psi - \phi_R) \\
    \partial_T \pi &= -\partial_R\phi_R - \frac{2}{R}\phi_R  + F\psi .
\end{align*}
$$

Přepíšeme systémdo souřadnic $(t,\ r)$, kde $t$ je hyperboloidní čas a $r$ je compactified radius. Transformace je dána jako

$$
T = t + H(R), \quad R = R(r),
$$
kde $r\in [0,r_{\!\tiny\mathscr{I}}]$ je kompaktifikovaný poloměr a $H(R)$ je height function. Dále definujeme $R' \equiv \frac{dR}{dr}$.

Jako kompaktifikaci volíme (Gautam et al. 2021, rov. (3))
$$
R(r) = \frac{r}{\Omega(r)^{1/(1-n)}}
     = r\left(1 - \frac{r^2}{r_{\!\tiny\mathscr{I}}^2}\right)^{1/(1-n)},
\qquad 1 < n \le 2,\quad r\in[0,r_{\!\tiny\mathscr{I}}],
$$
takže $R' \sim R^n$ pro $r \to r_{\!\tiny\mathscr{I}}$. (Paper později přechází na mírně odlišnou kompaktifikaci vhodnější pro časovou integraci; tou se zatím nezabýváme a držíme se rov. (3).)

Height function $H$ pak volíme podmínkou $R'(1 - H') = 1$, tj. $H' = 1 - 1/R'$. Tím dostáváme odchozí radiální rychlost světla $c^r_+ = 1$ identicky (odchozí pulzy se rozliší dokonale) a příchozí $c^r_- = -1/(2R' - 1)$, která má v počátku hodnotu $-1$ a monotónně klesá k $0$ na $\mathscr{I}^+$.

Podle dalších transformací dostáváme čtyři varianty rovnic, které se liší regularizací a charakteristickými proměnnými.

## Systémy rovnic
### Neregularizovaný 

$$
\begin{align*}
    \partial_t \psi &= -\pi, \\[0.3em]
    \partial_t \phi_R
    &= -\frac{R'}{2R' - 1}\,\partial_r \pi
       - \frac{R' - 1}{2R' - 1}
       \left(
           \partial_r + \frac{2R'}{R}
       \right)\phi_R  \\
    &\quad
       + \gamma_2 \frac{R'}{2R' - 1}
       \left[
           \partial_r \psi + (R' - 1)\pi - R'\phi_R
       \right]
       + \frac{R'(R' - 1)}{2R' - 1}F\psi, \\[0.3em]
    \partial_t \pi
    &= -\frac{R' - 1}{2R' - 1}\,\partial_r \pi
       - \frac{R'}{2R' - 1}
       \left(
           \partial_r + \frac{2R'}{R}
       \right)\phi_R \\
    &\quad
       + \gamma_2 \frac{R' - 1}{2R' - 1}
       \left[
           \partial_r \psi + (R' - 1)\pi - R'\phi_R
       \right]
       + \frac{(R')^2}{2R' - 1}F\psi .
\end{align*}
$$

### Neregularizovaný outgoing ingoing

Zavádíme příchozí a odchozí charakteristické proměnné
$$
    \sigma^+ \equiv -\pi + \phi_R,
    \qquad
    \sigma^- \equiv -\pi - \phi_R .
$$

$$
\begin{align*}
    \partial_t \psi
    &= \frac{1}{2}\left(\sigma^+ + \sigma^-\right), \\[0.3em]
    \partial_t \sigma^+
    &= \frac{1}{2R' - 1}
       \left[
           \partial_r \sigma^+
           + \frac{R'}{R}\left(\sigma^+ - \sigma^-\right)
           - R'F\psi
       \right] \\
    &\quad
       + \gamma_2
       \left[
           \frac{1}{2R' - 1}
           \left(
               \partial_r \psi + \frac{\sigma^-}{2}
           \right)
           - \frac{\sigma^+}{2}
       \right], \\[0.3em]
    \partial_t \sigma^-
    &= -\left[
           \partial_r \sigma^-
           + \frac{R'}{R}\left(\sigma^- - \sigma^+\right)
           + R'F\psi
       \right] \\
    &\quad
       - \gamma_2
       \left[
           \partial_r \psi
           + \frac{\sigma^-}{2}
           - (2R' - 1)\frac{\sigma^+}{2}
       \right].
\end{align*}
$$

### Regularizovaný

Zavádíme regularizované proměnné
$$
    \tilde{\psi} \equiv \chi\psi,
    \qquad
    \tilde{\phi}_R \equiv \chi\phi_R + \chi'\psi,
    \qquad
    \tilde{\pi} \equiv \chi\pi,
    \qquad
    \chi \equiv \sqrt{1 + R^2}.
$$

$$
\begin{align*}
    \partial_t \tilde{\psi}
    &= -\tilde{\pi}, \\[0.3em]
    \partial_t \tilde{\phi}_R
    &= -\frac{R'}{2R' - 1}\,\partial_r \tilde{\pi}
       - \frac{R' - 1}{2R' - 1}\,\partial_r \tilde{\phi}_R
       - \frac{2R'(R' - 1)}{(2R' - 1)\chi^2R}\,\tilde{\phi}_R \\
    &\quad
       + \frac{R'(R' - 1)}{2R' - 1}
       \left(
           \frac{3}{\chi^4} + F
       \right)\tilde{\psi} \\
    &\quad
       + \gamma_2\frac{R'}{2R' - 1}
       \left[
           \partial_r \tilde{\psi}
           + (R' - 1)\tilde{\pi}
           - R'\tilde{\phi}_R
       \right], \\[0.3em]
    \partial_t \tilde{\pi}
    &= -\frac{R' - 1}{2R' - 1}\,\partial_r \tilde{\pi}
       - \frac{R'}{2R' - 1}\,\partial_r \tilde{\phi}_R
       - \frac{2(R')^2}{(2R' - 1)\chi^2R}\,\tilde{\phi}_R \\
    &\quad
       + \frac{(R')^2}{2R' - 1}
       \left(
           \frac{3}{\chi^4} + F
       \right)\tilde{\psi} \\
    &\quad
       + \gamma_2\frac{R' - 1}{2R' - 1}
       \left[
           \partial_r \tilde{\psi}
           + (R' - 1)\tilde{\pi}
           - R'\tilde{\phi}_R
       \right].
\end{align*}
$$

### Regularizovaný outgoing ingoing

Zavádíme regularizované charakteristické proměnné
$$
    \tilde{\psi} \equiv \chi\psi,
    \qquad
    \tilde{\sigma}^+ \equiv \chi^2\sigma^+,
    \qquad
    \tilde{\sigma}^- \equiv \chi\sigma^- .
$$

$$
\begin{align*}
    \partial_t \tilde{\psi}
    &= \frac{1}{2}
       \left(
           \frac{\tilde{\sigma}^+}{\chi}
           + \tilde{\sigma}^-
       \right), \\[0.3em]
    \partial_t \tilde{\sigma}^+
    &= \frac{1}{2R' - 1}
       \left[
           \partial_r \tilde{\sigma}^+
           - \frac{2RR'}{\chi^2}\tilde{\sigma}^+
           + \frac{R'}{R}
             \left(
                 \tilde{\sigma}^+ - \chi\tilde{\sigma}^-
             \right)
           - \frac{R'}{\chi}F\tilde{\psi}
       \right] \\
    &\quad
       + \gamma_2
       \left[
           \frac{1}{2R' - 1}
           \left(
               \chi\partial_r \tilde{\psi}
               - \frac{RR'}{\chi}\tilde{\psi}
               + \chi\frac{\tilde{\sigma}^-}{2}
           \right)
           - \frac{\tilde{\sigma}^+}{2}
       \right], \\[0.3em]
    \partial_t \tilde{\sigma}^-
    &= -\partial_r \tilde{\sigma}^-
       + \left(
           \frac{RR'}{\chi^2} - \frac{R'}{R}
       \right)\tilde{\sigma}^-
       + \frac{R'}{R\chi}\tilde{\sigma}^+
       - R'F\tilde{\psi} \\
    &\quad
       - \gamma_2
       \left[
           \partial_r \tilde{\psi}
           - \frac{RR'}{\chi^2}\tilde{\psi}
           + \frac{\tilde{\sigma}^-}{2}
           - (2R' - 1)\frac{\tilde{\sigma}^+}{2\chi}
       \right].
\end{align*}
$$

# Okrajové (edge) tvary rovnic

Bulk rovnice výše obsahují členy singulární ve dvou bodech domény: v počátku $r=0$ (členy $\propto 1/R$) a na $\mathscr{I}^+$, tj. $r = r_{\!\tiny\mathscr{I}}$ (koeficienty s $R'$, které divergují, neboť $R\to\infty$). V těchto dvou mřížkových bodech nahrazujeme bulk rovnice jejich limitními tvary, které jsou konečné. Tím je RHS dobře definovaná na celé doméně.

## Operátor $\tilde{\partial}_r$

Aby se v rovnicích neobjevoval samostatný singulární člen $2R'/R$, zavádíme (Gautam et al. 2021, rov. (19)–(20)) operátor
$$
\tilde{\partial}_r \varphi
\equiv \chi^2\left(\partial_r + \frac{2R'}{R}\right)\frac{\varphi}{\chi^2}
= \partial_r \varphi + \frac{2R'}{(1+R^2)R}\,\varphi
= \partial_r \varphi + \frac{2R'}{R\,\chi^2}\,\varphi .
$$
Koeficient $2R'/(R\chi^2)$ je na rozdíl od $2R'/R$ konečný: pro $r\to r_{\!\tiny\mathscr{I}}$ jde k nule (jako $\sim R^{\,n-3}$). Na $\mathscr{I}^+$ tedy $\tilde{\partial}_r \to \partial_r$.

## Počátek $r = 0$

V počátku není fyzikální hranice, jen souřadnicová singularita sférických souřadnic. Regularita vynucuje paritu polí: $\psi$ a $\pi$ jsou sudé, $\phi_R$ liché (a tedy pro charakteristické proměnné $\sigma^+(-r) = \sigma^-(r)$). Singulární člen $\phi_R/R$ řešíme l'Hôpitalovým pravidlem, $\phi_R/R \to \partial_r\phi_R / R'$, přičemž $R' = 1$ v počátku. Tím $-\partial_r\phi_R - 2\phi_R/R \to -3\,\partial_r\phi_R$. Výsledné rovnice platí shodně ve fyzikálních i hyperboloidních souřadnicích.

Systém $(\psi, \phi_R, \pi)$ — Gautam et al. 2021, rov. (9):
$$
\begin{align*}
    \partial_t \psi(t,0) &= -\pi(t,0), \\[0.3em]
    0 &= -\partial_r \pi(t,0), \\[0.3em]
    \partial_t \pi(t,0) &= -3\,\partial_r \phi_R(t,0) + F\psi(t,0).
\end{align*}
$$

Charakteristický systém $(\psi, \sigma^+, \sigma^-)$ — Gautam et al. 2021, rov. (14):
$$
\begin{align*}
    \partial_t \psi(t,0) &= \sigma^+(t,0) = \sigma^-(t,0), \\[0.3em]
    \partial_t \sigma^+(t,0) &= -3\,\partial_r \sigma^-(t,0) - F\psi(t,0), \\[0.3em]
    \partial_t \sigma^-(t,0) &= -3\,\partial_r \sigma^-(t,0) - F\psi(t,0) = \partial_t \sigma^+(t,0).
\end{align*}
$$

Pro **regularizovaný charakteristický** systém $(\tilde\psi, \tilde\sigma^+, \tilde\sigma^-)$ platí v počátku $\chi = 1$, $\chi' = 0$, tedy $\tilde\psi = \psi$, $\tilde\sigma^\pm = \sigma^\pm$ a rovněž $\partial_r \tilde\sigma^\pm = \partial_r \sigma^\pm$. Edge tvar v počátku je proto doslova rov. (14) s vlnkovanými proměnnými.

> **Pozor:** pro regularizovaný systém $(\tilde\psi, \tilde\phi_R, \tilde\pi)$ tato náhrada neplatí — v počátku je $\partial_r \tilde\phi_R = \partial_r \phi_R + \psi$ (protože $\mathrm{d}\chi'/\mathrm{d}r = 1$ v $r=0$), takže rov. (9) s vlnkami by byla nesprávná. Paper edge tvary pro tento systém netabuluje; pro evoluci sahající až na $\mathscr{I}^+$ je proto vhodná charakteristická verze.

## Future null infinity $\mathscr{I}^+$ ($r = r_{\!\tiny\mathscr{I}}$)

Zde $R\to\infty$ a $R'\to\infty$. Limitní tvar se získá z bulk rovnic v limitě $R'\to\infty$: prefaktor $1/(2R'-1)\to 0$ anuluje část pravé strany, zatímco ostatní členy se skládají do konečných limit (např. $1/(2R'-1)\cdot(R'/R)(\tilde\sigma^+ - \chi\tilde\sigma^-) \to -\tilde\sigma^-/2$), a $\tilde{\partial}_r \to \partial_r$. Paper tuto limitu odvozuje pouze pro charakteristické proměnné, jejichž reškálování je „ostřejší" a singulární členy se v něm ruší.

Regularizovaný charakteristický systém $(\tilde\psi, \tilde\sigma^+, \tilde\sigma^-)$, s $\gamma_2 \simeq 1/R$ poblíž $\mathscr{I}^+$ — Gautam et al. 2021, rov. (23):
$$
\begin{align*}
    \partial_t \tilde\psi &= \frac{\tilde\sigma^-}{2}, \\[0.3em]
    \partial_t \tilde\sigma^+ &= -\frac{\tilde\sigma^-}{2} - \frac{\chi F \tilde\psi}{2}, \\[0.3em]
    \partial_t \tilde\sigma^-
    &= -\left[
            \left(\partial_r + \frac{\tilde{\partial}_r}{2}\right)\tilde\sigma^-
            - \frac{R'}{\chi^2}\tilde\sigma^+
            + R'F\tilde\psi
        \right] \\
    &\quad
        - \gamma_2\left[
            -\frac{R'}{\chi}\tilde\psi
            - (2R'-1)\frac{\tilde\sigma^+}{2\chi}
        \right].
\end{align*}
$$

# Rezidua

Nechť $\Psi(x,t)$ je přesné analytické řešení libovolné rovnice $(\star)$. 
Nechť $\psi^i_{\Delta x}(t)$ je časovou evolucí počátečních dat  $\psi^i_{\Delta x}(0) = \Psi(x_i,0)$ na gridu s diskretizací $\Delta x$ pomocí diskretizované verze rovnice $(\star)$.

Definujme reziduum v čase $t$ pro diskretizaci $\Delta x$ jako
$$
R^{i}_{\Delta x, \xi}(t) \equiv |\psi^i_{\Delta x}(t) - \psi^i_{\frac{\Delta x}{\xi}}(t)|,\quad \xi > 1.
$$

Někdy se hodí mít také fourierovské reziduum, které je definováno jako diskrétní Fourierova transformace rezidua v prostorovém indexu $i$.