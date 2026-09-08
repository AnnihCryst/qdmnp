# Зависимости для теоретической статьи

В этом каталоге находятся одиннадцать пар программ:

```text
расчётный скрипт -> самодостаточный NPZ -> скрипт построения -> рисунок
```

Расчётный скрипт выполняет интегрирование и сохраняет использованные входные
параметры, физические константы, табличную дисперсию золота, параметры
рациональной аппроксимации, рассчитанные массивы, единицы измерения и
диагностики сходимости. Скрипт построения читает только NPZ, поэтому изменение
стиля рисунка не требует повторного тяжёлого расчёта.

## 1. Возбуждение экситона как функция флюенса

- `qd_mnp_calculate_excitation_fluence.py` — расчёт;
- `qd_mnp_plot_excitation_fluence.py` — построение.

Вычисляется

\[
P_{\mathrm{exc}}(\mathcal F)=\rho_{ee}(t_{\mathrm{read}};\mathcal F).
\]

Эта зависимость показывает, усиливает или подавляет МНЧ возбуждение экситона,
насколько изменяется требуемый флюенс и какая комбинация положения КТ и
поляризации наиболее эффективна. Сохраняются изолированная КТ (`bare_qd`) и
пять неэквивалентных каналов осесимметричного сфероида.

Сравнение совместимых NPZ с `material_fit_modes=1` и
`material_fit_modes=9` показывает влияние точности описания частотной
дисперсии золота. Это не сравнение пространственных моделей взаимодействия.

## 2. Оптическая работа как функция флюенса

- `qd_mnp_calculate_work_loss_fluence.py` — расчёт;
- `qd_mnp_plot_work_loss_fluence.py` — построение.

Сохраняются две разные величины:

\[
\sigma_{\mathrm{QS,work}}(E_L;\mathcal F)
=\frac{k(E_L)}{\varepsilon_0}\operatorname{Im}\alpha_{\mathrm{eff}}(E_L;\mathcal F)
\]

и

\[
\sigma_{\mathrm{energy}}(\mathcal F)=
\frac{\int E_{\mathrm{inc}}(t)\,d\mu_{\mathrm{total}}(t)/dt\,dt}{\mathcal F}.
\]

Они показывают, сопровождается ли изменение населённости измеримым изменением
оптического отклика и работы внешнего поля. Для гибридной системы сохраняются
также отклик изолированной МНЧ (`bare_mnp`) и разность «гибрид минус bare-MNP».
Эти величины нельзя без отдельного энергетического разложения называть чистым
поглощением металла.

## 3. Временная динамика населённости

- `qd_mnp_calculate_population_dynamics.py` — расчёт;
- `qd_mnp_plot_population_dynamics.py` — построение.

Вычисляется зависимость

\[
\rho_{ee}(t)
\]

при фиксированном импульсе. Она показывает момент формирования усиления или
подавления, Rabi-like осцилляции, максимальную и остаточную населённость.
Всегда сохраняется контрольная динамика изолированной КТ (`bare_qd`). Для
минимального рисунка после анализа `P_exc(F)` выбираются оптимальный и один
контрастный гибридные каналы.

## Пространственные конфигурации

- `axis_long` — КТ у вершины, поле вдоль оси;
- `axis_trans` — КТ у вершины, поле поперёк оси;
- `side_long` — КТ у боковой поверхности, поле вдоль оси;
- `side_trans_radial` — КТ сбоку, поперечное поле радиально;
- `side_trans_tangential` — КТ сбоку, поперечное поле тангенциально.

## 4. Практические метрики DD/FQS как функции зазора

Пять новых пар программ отвечают непосредственно на вопросы об усилении,
пороговой интенсивности, сдвиге, эффективном уширении, положении КТ и
направлении электрического поля:

| Зависимость | Расчёт | Построение | Физический вопрос |
|---|---|---|---|
| `G_exc(g)` | `qd_mnp_calculate_excitation_gain_gap.py` | `qd_mnp_plot_excitation_gain_gap.py` | Усиливает или подавляет МНЧ резонансное возбуждение КТ? |
| `(E_res(g)-E_res,0)/Gamma0` | `qd_mnp_calculate_resonance_shift_gap.py` | `qd_mnp_plot_resonance_shift_gap.py` | Насколько смещается максимум возбуждения? |
| `Gamma_eff(g)/Gamma0` | `qd_mnp_calculate_spectral_width_gap.py` | `qd_mnp_plot_spectral_width_gap.py` | Насколько меняется рабочее спектральное окно возбуждения? |
| `F_eta(g)/F_eta,0` | `qd_mnp_calculate_threshold_fluence_gap.py` | `qd_mnp_plot_threshold_fluence_gap.py` | Во сколько раз меняется требуемый флюенс и, при одинаковом импульсе, пиковая интенсивность? |
| `delta_spec(g)` и `delta_F(g)` | `qd_mnp_calculate_model_discrepancy_gap.py` | `qd_mnp_plot_model_discrepancy_gap.py` | Начиная с какого расстояния DD воспроизводит FQS с заданной точностью? |

Здесь `g` — расстояние поверхность–поверхность, а не расстояние между
центрами. Для вершины `R=c+r_QD+g`, для боковой поверхности
`R=a+r_QD+g`.

### Определения метрик

Для общего QD-селективного спектра `S(E,g)` сохраняются

\[
G_{\rm exc}^{\rm opt}(g)=
\frac{\max_E S(E,g)}{\max_E S_0(E)},
\qquad
G_{\rm exc}^{*}(g)=\frac{S(E_*,g)}{S_0(E_*)}.
\]

`S` можно вычислять двумя способами:

- `--spectral-observable linear_qd_response` — быстрый слабополевой
  монохроматический показатель `|p_QD/E_inc|^2`;
- `--spectral-observable weak_pulse_excitation` — более дорогой операционный
  короткоимпульсный спектр `rho_ee(t_read)/F` при малом флюенсе.

Резонансом считается пик, отслеживаемый около энергии изолированной КТ в
заранее фиксированном окне. `Gamma0` — FWHM изолированной КТ, извлечённая тем
же алгоритмом. `Gamma_eff` — ширина связной области на половине prominence
выбранного максимума. Если обнаружен сравнимый второй пик, ширина получает
статус `split_or_ambiguous` и не выдаётся как одно число. Для импульсного
спектра это операционная ширина возбуждения, включающая спектральную ширину
импульса и power broadening, а не рассчитанное изменение времени жизни КТ.

Порог определяется по первой возрастающей ветви до первого максимума Раби:

\[
\mathcal F_\eta=\inf\{\mathcal F:\rho_{ee}(t_{\rm read})\ge\eta\}.
\]

Основная сохраняемая величина `F_eta/F_eta,0` меньше единицы при выигрыше.
Одновременно сохраняется обратная эффективность `F_eta,0/F_eta`, абсолютный
порог, соответствующая пиковая интенсивность и ошибка DD относительно FQS

\[
D_{\mathcal F}=\frac{\mathcal F_{\eta,DD}-
\mathcal F_{\eta,FQS}}{\mathcal F_{\eta,FQS}}.
\]

Левое/правое цензурирование и недостижение порога на первой Rabi-ветви
сохраняются отдельными статусами; из таких точек отношение порогов не
формируется.

Спектральное расхождение вычисляется на общей сетке, без индивидуальной
нормировки кривых:

\[
\delta_{\rm spec}(g)=
\left[
\frac{\int_W [S_{DD}(E,g)-S_{FQS}(E,g)]^2\,dE}
     {\int_W S_{FQS}^2(E,g)\,dE}
\right]^{1/2}.
\]

Граница применимости DD — первый зазор, после которого допуск выполняется во
всех последующих, более далёких, точках, а не первое случайное пересечение.

### Математические ядра и одинаковость сценария

Новые расчёты не копируют уравнения взаимодействия. Они вызывают действующие
API проекта:

- DD во времени — `HybridQDPlasmonModel.solve`;
- DD в частотной области — `LegacyDipoleInteraction.frequency_response`;
- FQS во времени — `FullQSSpheroidPulseModel.solve`;
- FQS в частотной области — аналитические axial/equatorial spheroidal Green
  kernels и `solve_linear_hybrid_response`.

Для DD и FQS одинаковы геометрия, материал золота, параметры КТ, импульс,
энергетическая/флюенсная сетка и момент чтения. В частотном режиме обе ветви
используют одну и ту же прямую табличную дисперсию материала, поэтому
`delta_spec` изолирует пространственное приближение. В импульсном режиме обе
ветви используют одну и ту же причинную Lorentz/ADE-аппроксимацию материала.

Направление распространения луча и волновой вектор не вводятся. Обозначения
`long`, `trans`, `radial`, `tangential` относятся только к направлению вектора
электрического поля относительно оси и локальной поверхности сфероида.

### Повторное использование тяжёлого спектрального расчёта

Первый из четырёх спектральных расчётов сохраняет весь массив
`S(model,channel,gap,energy)`, комплексные `A/B/K`-отклики, исходный материал,
геометрию, параметры и диагностики. Остальные метрики можно получить из него
без повторного решения моделей.

При выборе `weak_pulse_excitation` дополнительно сохраняются фактический общий
интервал интегрирования, амплитуды и пиковые интенсивности обоих проверочных
флюенсов, коэффициенты причинной Lorentz/ADE-аппроксимации и сертификаты FQS.

Пример:

```powershell
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_excitation_gain_gap --preset publication --gamma2-coherence-mev $articleGamma2MeV --d-debye $articleDDebye --output results/article/excitation_gain_gap.npz

.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_resonance_shift_gap --source-artifact results/article/excitation_gain_gap.npz --output results/article/resonance_shift_gap.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_spectral_width_gap --source-artifact results/article/excitation_gain_gap.npz --output results/article/spectral_width_gap.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_model_discrepancy_gap --source-artifact results/article/excitation_gain_gap.npz --output results/article/model_discrepancy_gap.npz

.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_excitation_gain_gap results/article/excitation_gain_gap.npz --output results/article/excitation_gain_gap.png
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_resonance_shift_gap results/article/resonance_shift_gap.npz --output results/article/resonance_shift_gap.png
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_spectral_width_gap results/article/spectral_width_gap.npz --output results/article/spectral_width_gap.png
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_model_discrepancy_gap results/article/model_discrepancy_gap.npz --dd-tolerance 0.1 --output results/article/model_discrepancy_gap.png
```

Нелинейный порог требует собственного расчёта:

```powershell
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_threshold_fluence_gap --preset publication --gamma2-coherence-mev $articleGamma2MeV --d-debye $articleDDebye --target-population 0.5 --output results/article/threshold_fluence_gap.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_threshold_fluence_gap results/article/threshold_fluence_gap.npz --output results/article/threshold_fluence_gap.png

# Не решая ОДУ повторно, выделить и построить delta_F из порогового артефакта
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_model_discrepancy_gap --source-artifact results/article/threshold_fluence_gap.npz --output results/article/threshold_model_discrepancy_gap.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_model_discrepancy_gap results/article/threshold_model_discrepancy_gap.npz --dd-tolerance 0.1 --output results/article/threshold_model_discrepancy_gap.png
```

## 5. Один осциллятор и многоосцилляторная дисперсия золота

Три специальные пары отделяют ошибку описания частотной дисперсии материала
от ошибки пространственной DD-модели:

| Зависимость | Расчёт | Построение | Что сравнивается |
|---|---|---|---|
| `alpha(E)` и `1/alpha(E)` | `qd_mnp_calculate_material_dispersion_comparison.py` | `qd_mnp_plot_material_dispersion_comparison.py` | Прямая табличная поляризуемость сфероида, её `N=1` и `N>=2` причинные лоренцевы аппроксимации |
| `S(E)=abs(p_QD/E_inc)^2` | `qd_mnp_calculate_excitation_spectrum_material_comparison.py` | `qd_mnp_plot_excitation_spectrum_material_comparison.py` | Последствия трёх представлений материала для слабополевого возбуждения КТ при одном и том же полном QS-ядре |
| `P_exc(F)=rho_ee(t_read;F)` | `qd_mnp_calculate_excitation_fluence_material_comparison.py` | `qd_mnp_plot_excitation_fluence_material_comparison.py` | Нелинейная импульсная динамика для `N=1` и производственной многоосцилляторной реализации при полностью одинаковом сценарии |

В первых двух расчётах ветвь `direct` использует интерполированные табличные
оптические константы золота непосредственно в локально-квазистатической модели.
Она является эталоном аппроксимации внутри этой модели, но не абсолютной
экспериментальной истиной. Временной `direct`-ветви в третьем расчёте нет:
произвольная табличная функция сама по себе не задаёт конечную причинную систему
ODE/ADE. Поэтому нелинейный график сравнивает `N=1` с `N=9`, а качество `N=9`
предварительно проверяется двумя частотными графиками относительно `direct`.

Используемые «осцилляторы» — материальные полюса

\[
\alpha_{\rm fit}(\omega)=\alpha_\infty+
\sum_{s=1}^{N}\frac{f_s}{\omega_s^2-\omega^2-i\gamma_s\omega},
\]

а не пространственные сфероидальные моды. Число пространственных членов
фиксируется отдельно (`spatial_order_max`) и одинаково для сравниваемых ветвей.
Проектная ветвь `N=1` не отождествляется автоматически с эффективным ярким
осциллятором Shah: здесь это намеренно грубая однополюсная аппроксимация той же
табличной дисперсии золота.

Быстрый сквозной пример:

```powershell
# 1. Ошибка материальной аппроксимации и положение/ширина LSPR
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_material_dispersion_comparison --output results/quick/material_dispersion_comparison.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_material_dispersion_comparison results/quick/material_dispersion_comparison.npz --output results/quick/material_dispersion_comparison.png

# 2. Слабополевой спектр КТ при одном полном QS-пространственном ядре
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_excitation_spectrum_material_comparison --preset quick --output results/quick/excitation_spectrum_material_comparison.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_excitation_spectrum_material_comparison --input results/quick/excitation_spectrum_material_comparison.npz --output results/quick/excitation_spectrum_material_comparison.png --allow-unconverged

# 3. Нелинейная населённость при одинаковых импульсе, сетке и t_read
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_excitation_fluence_material_comparison --preset quick --output results/quick/excitation_fluence_material_comparison.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_excitation_fluence_material_comparison results/quick/excitation_fluence_material_comparison.npz --allow-unconverged --output results/quick/excitation_fluence_material_comparison.png
```

`quick` проверяет только работоспособность цепочки и закономерно может не пройти
пространственный, спектральный или флюенсный сертификат. Режимы построения
`--allow-unconverged` разрешают такой рисунок только с водяным знаком. Для
данных статьи следует использовать `--preset publication`, задать одни и те же
экспериментально обоснованные параметры КТ/МНЧ и не ослаблять производственные
gates. Все три NPZ сохраняют исходную таблицу материала, фактически
использованные входы и константы, коэффициенты полюсов, абсолютные кривые,
остатки относительно `direct` там, где он определён, производные метрики и
диагностические сертификаты. Plotter-ы читают только эти NPZ.

Plotter может без ОДУ изменить окно спектральной особенности, фиксированную
энергию/тип gain, целевую населённость (с интерполяцией по сохранённой сетке)
и допуск DD. Производственный расчёт должен явно задавать экспериментально
обоснованные `d`, `gamma1`, `Gamma2`; значения по умолчанию не являются
автоматически параметрами конкретной коллоидной КТ.

## Что именно сравнивают старые три пары программ

Все три расчётных скрипта используют для связанной системы только полный
аналитический локально-квазистатический отклик сфероида
`FullQSSpheroidPulseModel`. Контроли `bare_qd` и `bare_mnp` являются
изолированными объектами, а не диполь-дипольной моделью гибрида.

`material_fit_modes=1` означает один лоренцев осциллятор аппроксимации
дисперсии золота. Пространственный порядок задаётся независимо параметром
`spatial_order_max`; даже `spatial_order_max=1` не тождественен legacy-модели
точечного диполя МНЧ при конечном расстоянии.

Следовательно, старые три пары исследуют `P_exc(F)`, оптическую работу и
`rho_ee(t)` внутри full-QS модели. Прямое сравнение в них самих

\[
X_{\mathrm{dipole-dipole}}\quad\text{и}\quad X_{\mathrm{full-QS}}
\]

при одинаковых параметрах не реализовано. Эту задачу выполняют новые пять пар
из раздела 4, прежде всего `model_discrepancy_gap` и сохранённая в пороговом
артефакте `absolute_threshold_discrepancy_dd_vs_fqs`.

Запускать программы следует из корня репозитория, например:

```powershell
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_calculate_excitation_fluence --preset quick --output results/quick/excitation_fluence.npz
.\.venv\Scripts\python.exe -m article_observables.qd_mnp_plot_excitation_fluence results/quick/excitation_fluence.npz --allow-unconverged --output results/quick/excitation_fluence.png
```
