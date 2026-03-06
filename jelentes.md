# Neurális hálózat alapú emlőrák klasszifikáció — Jelentés

## 1. Bevezetés

Az emlőrák a leggyakoribb daganatos megbetegedés nők körében világszerte. A WHO adatai szerint évente több mint 2 millió új esetet diagnosztizálnak, és a korai felismerés jelentősen javítja a túlélési esélyeket. A mesterséges intelligencia és a gépi tanulás egyre fontosabb szerepet tölt be az orvosi diagnosztikában, különösen a képalkotó és szövettani vizsgálatok kiértékelésében, ahol a számítógépes rendszerek segíthetnek az orvosoknak a gyorsabb és pontosabb döntéshozatalban. Jelen munka célja egy egyszerű neurális hálózat alapú klasszifikációs modell felépítése és értékelése a Wisconsin Breast Cancer Diagnostic adatbázison, amely a szövettani minták digitalizált jellemzőit tartalmazza.

## 2. Adathalmaz és előfeldolgozás

A Wisconsin Breast Cancer Diagnostic adatbázis 569 szövettani mintát tartalmaz, amelyeket benignus (jóindulatú, 357 minta) vagy malignus (rosszindulatú, 212 minta) kategóriába soroltak. Minden mintához 30 numerikus jellemző tartozik, amelyeket a sejtek digitalizált képéből számítottak ki. Ezek a jellemzők három csoportba oszthatók: átlagértékek (mean), standard hibák (SE) és legrosszabb értékek (worst) — mindhárom csoportban 10-10 jellemzővel, mint például sugár, textúra, kerület, terület, simaság, kompaktság, konkávitás és szimmetria.

Az adathalmaz nem tartalmazott hiányzó értékeket, ami leegyszerűsítette az előfeldolgozást. Az osztályeloszlás enyhe egyensúlytalanságot mutatott (benignus/malignus arány: 1,68:1), de ez nem volt olyan mértékű, hogy speciális kezelést igényeljen.

Az adatokat 80/20 arányban osztottuk fel tanító és teszt halmazra, stratifikált mintavételezéssel, amely biztosította az osztályarányok megőrzését mindkét halmazban. A jellemzők skálázásához StandardScaler-t alkalmaztunk, amely minden jellemzőt nulla átlagúra és egységnyi szórásúra transzformált. Fontos, hogy a skálázó paramétereit kizárólag a tanító halmazon illesztettük, majd ezeket alkalmaztuk a teszt halmazra is — ezzel elkerülve az adatszivárgást (data leakage).

## 3. Modell architektúra

A modell egy egyszerű előrecsatolt (feedforward) neurális hálózat, amelyet PyTorch keretrendszerben implementáltunk. Az architektúra a következő:

- **Bemeneti réteg**: 30 neuron (a 30 jellemzőnek megfelelően)
- **Első rejtett réteg**: 64 neuron, ReLU aktivációs függvénnyel
- **Második rejtett réteg**: 32 neuron, ReLU aktivációs függvénnyel
- **Kimeneti réteg**: 1 neuron, Sigmoid aktivációs függvénnyel

A modell összesen 4097 tanítható paramétert tartalmaz. A ReLU (Rectified Linear Unit) aktivációs függvényt választottuk a rejtett rétegekhez, mivel egyszerű, hatékony és elkerüli a gradiens-elhalás problémáját. A kimeneti Sigmoid függvény a végeredményt 0 és 1 közötti valószínűségre szorítja, amely a benignus diagnózis valószínűségeként értelmezhető.

A tanítás 100 epoch-on keresztül zajlott, Adam optimalizálóval (tanulási ráta: 0,001) és bináris keresztentrópia (BCE) veszteségfüggvénnyel, amely a standard választás bináris klasszifikációs feladatokhoz.

## 4. Eredmények

### 4.1 Alapvető teljesítménymutatók

A modell a teszt halmazon 95,6%-os pontosságot (accuracy) ért el. A részletes konfúziós mátrix a 114 teszt mintán:

|                | Becsült malignus | Becsült benignus |
|----------------|:----------------:|:----------------:|
| Valós malignus |     41 (TN)      |      1 (FN)      |
| Valós benignus |      4 (FP)      |     68 (TP)      |

A klasszifikációs metrikák:
- **Malignus precision**: 0,91 — a malignus predikciók 91%-a helyes
- **Malignus recall (szenzitivitás)**: 0,976 — a valós malignus esetek 97,6%-át felismerte
- **Benignus recall (specificitás)**: 0,944 — a valós benignus esetek 94,4%-át felismerte
- **F1-score**: 0,94 (malignus), 0,96 (benignus)

### 4.2 ROC-görbe és AUC

A ROC-görbe (Receiver Operating Characteristic) a modell teljesítményét mutatja különböző döntési küszöbök mellett. A görbe alatti terület (AUC) értéke 0,9964, ami közel tökéletes diszkriminációs képességet jelez. Összehasonlításképpen: egy véletlenszerű osztályozó AUC értéke 0,5 lenne.

### 4.3 Keresztvalidáció

Az 5-szörös stratifikált keresztvalidáció megerősítette az eredmények megbízhatóságát:

| Metrika   | Átlag  | Szórás |
|-----------|:------:|:------:|
| Accuracy  | 97,2%  | 1,16%  |
| AUC       | 99,6%  | 0,46%  |
| F1-score  | 97,8%  | 0,96%  |

Az alacsony szórásértékek azt mutatják, hogy a modell konzisztensen teljesít, függetlenül az adatfelosztástól. Ez fontos eredmény, mert kizárja, hogy az egyszeri felosztás „szerencsés" volt.

## 5. Hibásan klasszifikált esetek elemzése

A 114 teszt mintából 5 hibás predikció született. Ezek részletes vizsgálata a következő mintázatokat tárta fel:

**Hamis negatív eset (kihagyott rák)**: Egy malignus minta (53. minta) összes kulcsfontosságú jellemzője — sugár, textúra, kerület, terület, konkávitás — a benignus tartományba esett. A modell 82%-os biztonsággal benignusnak jelezte. Ez egy kis méretű, sima felszínű rosszindulatú tumor volt, amely a szövettani jellemzők alapján megtévesztően jóindulatúnak tűnt.

**Hamis pozitív esetek (4 eset)**: Ezek a minták szokatlanul nagy méretű benignus tumorok voltak (a sugár és kerület értékek a malignus tartományba estek), illetve egyes esetekben a textúra volt szokatlanul durva. A modell ezeket tévesen rosszindulatúnak jelezte.

Fontos megfigyelés, hogy minden hibásan klasszifikált minta a két osztály átfedési zónájában helyezkedik el a jellemzőtérben — ezek olyan „határesetek", amelyek bármely klasszifikátor számára nehézséget jelentenek.

## 6. Klinikai következmények

A modell teljesítménye ígéretes, de a klinikai alkalmazás szempontjából kritikus különbségek vannak a hibatípusok között:

**A hamis negatívok (kihagyott rákok) a legveszélyesebbek.** Ha egy rosszindulatú daganatot tévesen jóindulatúnak minősítünk, a beteg nem kap időben kezelést, ami a betegség progressziójához és rosszabb prognózishoz vezethet. Modellünkben 1 ilyen eset fordult elő a 42 malignus mintából (97,6%-os szenzitivitás).

**A hamis pozitívok kevésbé veszélyesek, de nem ártalmatlanok.** A tévesen rosszindulatúnak jelzett jóindulatú esetek felesleges további vizsgálatokhoz (biopszia, képalkotó vizsgálatok) és jelentős pszichológiai megterheléshez vezethetnek a beteg számára.

Klinikai környezetben érdemes lehet a döntési küszöböt 0,5-ről alacsonyabbra állítani — például 0,3-ra —, hogy a modell „óvatosabb" legyen és több mintát jelöljön potenciálisan rosszindulatúnak. Ez növelné a hamis pozitívok számát, de csökkentené a kihagyott rákok kockázatát, ami orvosi szempontból elfogadhatóbb kompromisszum.

## 7. Korlátok és jövőbeli irányok

A modellnek több korlátja van. Először, az adathalmaz viszonylag kicsi (569 minta), és egyetlen intézményből származik, ami korlátozza az általánosíthatóságot. Másodszor, az egyszerű feedforward architektúra nem feltétlenül ragadja meg a jellemzők közötti összetett, nemlineáris kapcsolatokat. Harmadszor, a jellemzők között erős korreláció áll fenn (pl. sugár-kerület-terület: r ≈ 0,99), amelynek kezelése — például főkomponens-elemzéssel (PCA) — javíthatná a modell hatékonyságát.

Jövőbeli fejlesztési lehetőségek közé tartozik: regularizáció (dropout rétegek) alkalmazása a túlillesztés csökkentésére, szisztematikus hiperparaméter-optimalizálás (grid search vagy Bayesian optimalizáció), összetettebb architektúrák kipróbálása, valamint a döntési küszöb optimalizálása a klinikai követelményekhez igazítva. Érdemes lenne továbbá a modellt nagyobb, többközpontú adathalmazon is validálni, hogy az általánosítóképességét felmérjük különböző populációk és mintavételi módszerek esetén.

## 8. Összefoglalás

A fejlesztett neurális hálózat 97,2%-os átlagos pontossággal és 0,996-os AUC értékkel képes megkülönböztetni a jóindulatú és rosszindulatú emlődaganatokat a Wisconsin adatbázison. A keresztvalidáció alacsony szórásértékei igazolták az eredmények konzisztenciáját és megbízhatóságát. A hibaelemzés feltárta, hogy a modell elsősorban a határesetekben téved — olyan tumoroknál, amelyek jellemzői a két osztály átfedési tartományába esnek, és amelyek valószínűleg emberi szakértők számára is kihívást jelentenének. A klinikai alkalmazás szempontjából fontos hangsúlyozni, hogy az ilyen rendszerek nem helyettesíthetik az orvosi szakvéleményt, hanem kiegészítő döntéstámogató eszközként szolgálhatnak a diagnosztikai folyamatban, segítve a patológusokat a kétes esetek kiszűrésében és a munkafolyamatok hatékonyságának növelésében.
