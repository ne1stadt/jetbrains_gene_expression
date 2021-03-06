# Задача 1:
Проведите анализ данных экспресии генов (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26728) :

* Информация о каком количестве генов есть в исследовании? Есть ли в данных дубликаты по генам? Что еще вы можете сказать о данных?
* Посчитайте средние значения экспресии каждого из генов в контроле, low_dose и high_dose. Если обнаружите дубликаты, усредните и их.
* Воспользуйтесь деревом решений высотой 2 и постройте классификаторы control vs low_dose, control vs high_dose. Какие гены оказались лучшими признаками для разделения? Совпали ли признаки в деревьях?
В качестве решения этой задачи приложите ссылку на открытый репозиторий в GitHub.

**Whole answer is in q1.py**



# Задача 2:
Ознакомьтесь с данными экспрессии генов: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26728

Ответьте на следующие вопросы:

* Что исследовалось?
* Какие дозы BPA использовались?
* Сколько особей использовалось?

**Answer:**
Исследовалось влияние Bisphenol A (пластик) на экспрессию разных генов мышей. Из-за сильной схожести генного транскриптома мыши и человека, можно с выокой точностью утверждать, что если BPA влияет на экспрессию генов у мышей, то повлияет и у человека. Мышей кормили едой с пластиком в разных дозах в течение 28 дней, делали эвтаназию и разнымим методоми мерили экспрессию разных генов. Исследование состоит из нескольких частей:

* Для 5 групп по 6 мышей (0, 5, 50, 500, 5000 ug/kg/d) измеряются основные показатели (изменение веса, инсулин, глюкоза и т. д.)
* С помощью Agilent Whole Mouse Genome microarrays (по принципу комплементарности, крашенные нуклеатиды "садятся" на заранее приготовленную пластнику, а затем измеряется интенсивность отражения - т. к. интенсивность пропорциональна количеству и, соответственно, экспресии). Таким образом можем найти гены, значения экспрессии у которых в контроле отличаются от измеряемых. Этот метод неточен, но необходим, поскольку проводить более точные иследования на соответствующем количестве генов очень сложно. На этом этапе ипользется три группы по 6 мышей: 0 ug/kg/d = control, 50 ug/kg/d = TDI = low_dose, 5000 ug/kg/d = NOAEL = high_dose.
* Гены, которые отличились в первом этапе исследуются дальше с помощью Real Time PCR (насколько я понял, это методика при которой в цепь вставляется комплиментарный цонд с красителем и заглушителем (silencer), который не позволяет детектировать сигнал. Когда полимераза доходит до этого зонда, она "разбирает его", разводя краситель и глушитель). Таким образом, в статье подтверждаются паттерны, полученные в первом методе. Исследовадось 5 групп по 6 мышей - 0, 5, 50, 500, 5000 ug/kg/d.
* Далее проводится исследование белка методом Western Blot (насколько я понял, под воздействием тока в геле белок перемещается по пленке и оставляет следы, интенсивность которых соответствует экспрессии). Тут также исследуются 5 групп по 6 мышей - 0, 5, 50, 500, 5000 ug/kg/d.
* В заключение печеночные липиды были покрашены красной краской и получена картина клетки при дозах 0, 50, 5000 ug/kg/d.

Таким образом влияние BPA было анализировано на многих уровнях: инсулина, мРНК, белка и липидов.
