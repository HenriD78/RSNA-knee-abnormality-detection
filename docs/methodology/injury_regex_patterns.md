# Injury extraction regex patterns

Extracted verbatim from `0-941-sota-rsna-knee-dinov2-radimagenet.ipynb`, cells 17–19 (counting from 1). Each regex below is the actual compiled pattern, with a single backslash for regex escapes. Python usage: `re.compile(r"pattern")`.

## How the notebook assigns targets

Reports are lowercased, transliterated for selected characters, stripped of combining accents and soft hyphens, and have underscores, hyphens, slashes and backslashes replaced by spaces. Whitespace is collapsed and the report is split into clauses. Apply these patterns to the normalized text.

- ACL, MCL, medial meniscus and lateral meniscus: match `ANAT[target]`, or match the corresponding anatomy stem with its direction within 55 characters. Require a match from `TEAR | DEGEN | INJURY` or a grade in the same clause; normal descriptions can supply negative evidence. Unsided plural menisci can supply evidence for both meniscus targets.
- Medial/lateral OA: require `OA_EVIDENCE` and `TF_SITE` with the corresponding side within 45 characters. PF OA uses `PF_SITE`. Unlocalized positive OA evidence can contribute to all three OA targets through the enabled inheritance rule.
- Effusion, synovitis, Baker's cyst, contusion and fracture: use `DIRECT[target]`. Fracture additionally includes `STEM_FRACTURE`. `DECOY` excludes specified misleading clauses.
- `PRE_NEG` before the finding or `POST_NEG` after it marks negative evidence within 90 characters. Contrast words stop preceding negation. `UNCERTAIN` is checked first and makes the clause uncertain. `NORMALITY` makes it negative unless `TEAR` or `GRADE_HIGH` is present.
- `SEV_LOW`, `SEV_HIGH` and grade patterns influence scores. Degenerative marrow context reduces contusion weight; trauma context increases it. Synovitis can receive a fallback score from effusion and `SYNOVIAL_PROXY`.

The extractor returns a score, confidence, and positive/negative evidence counts for each of 12 targets. These are heuristic scores; the regex matches do not establish extraction accuracy, and the extractor does not choose a single injury.

## Compiled patterns

### _SENT_SPLIT

```regex
(?<=[.;!?])\s+|\n+
```

### PRE_NEG

```regex
\bno\b|\bnot\b|\bwithout\b|\bnegative for\b|\babsence\b|\bno evidence\b|\bfree of\b|\bnone\b|\bneither\b|\bnor\b|\bsin\b|\bno hay\b|\bausencia\b|\bausentes?\b|\bno se\b|\bpas de\b|\bsans\b|\baucune?\b|\bgeen\b|\bzonder\b|\bniet\b|\bkeine?[nmrs]?\b|\bohne\b|\bnicht\b|\bkein\b|\bnema\b|\bbez\b|\bnisu\b|\bnije\b|\bδεν\b|\bχωρις\b|ουδεν|\bουτε\b|\bбез\b|\bне\b|липсва|\bняма\b
```

### POST_NEG

```regex
\byok\b|\byoktur\b|izlenmemekte|saptanmadi|\bdegil\b|gozlenmemekte|mevcut degil|eslik etmiyor|\bizlenmedi\b|izlenmemistir|saptanmamistir|gorulmemistir|\bnema znakova\b|bez znakova
```

### NEGATION

```regex
\bno\b|\bnot\b|\bwithout\b|\bnegative for\b|\babsence\b|\bno evidence\b|\bfree of\b|\bnone\b|\bneither\b|\bnor\b|\bsin\b|\bno hay\b|\bausencia\b|\bausentes?\b|\bno se\b|\bpas de\b|\bsans\b|\baucune?\b|\bgeen\b|\bzonder\b|\bniet\b|\bkeine?[nmrs]?\b|\bohne\b|\bnicht\b|\bkein\b|\bnema\b|\bbez\b|\bnisu\b|\bnije\b|\bδεν\b|\bχωρις\b|ουδεν|\bουτε\b|\bбез\b|\bне\b|липсва|\bняма\b|\byok\b|\byoktur\b|izlenmemekte|saptanmadi|\bdegil\b|gozlenmemekte|mevcut degil|eslik etmiyor|\bizlenmedi\b|izlenmemistir|saptanmamistir|gorulmemistir|\bnema znakova\b|bez znakova|\bunremarkable\b
```

### NORMALITY

```regex
\bnormal|\bintact\b|\bpreserved\b|\bwithin normal limits\b|limites normales|\bconservad|\bintegr|\bnormales\b|\bdoga(l|ll)\b|korunmus|\bnormaldir\b|olagan|\buredn|\bocuvan|\bodrzan|\bintakt|\bprimjeren|\bodrzanog kontinuiteta|\bodržan|φυσιολογικ|ακεραι|δεν παρατηρουνται|δεν σημειωνονται|unauffallig|regelrecht|\bo\.?b\.?\b|нормал|запазен|съхранен|\bбез особености\b|интактн|\bgaaf\b|\bnormaal\b
```

### NORMAL_PHRASE

```regex
\bsin alteracion|\bsin cambios\b|\bsin particularidad|\bsin hallazgos\b|\bsin lesion|\bsin signos de (rotura|lesion)|\bcontinu[oa]s?\b|\bcontinuidad conservada\b|\bno abnormalit|\bno significant abnormalit|\bunremarkable\b|\bno evidence of (tear|injury|abnormalit)|\bohne auffalligkeit|\bkein nachweis\b|\bohne befund\b|\bgeen afwijking|\bzonder afwijking|\bsans anomalie|\bpas d[e']anomalie|\bbez osobitosti\b|\bbez znakova (rupture|lezije)\b|\bbez patoloskih\b|χωρις αλλοιωσ|χωρις παθολογ|δεν παρατηρουνται (αξιολογα|παθολογ)|\bбез особености\b|\bбез патологич|\bбез данни за\b|\bozel bir ozellik yok|\bpatolojik bulgu (yok|izlenmemis)
```

### UNCERTAIN

```regex
\bpossible\b|\bprobable\b|\bsuspicious\b|\bsuspected?\b|cannot (be )?exclude|\bmay\b|\bquestionable\b|\bequivocal\b|\br/o\b|\bdd\b|\blikely\b|\bsuggest|\bcompatible with\b|\bposible\b|sin criterios categoricos|\bdudos|\bsugier|\bmuhtemel\b|\bolasi\b|\bsupheli\b|\bizlenim|\bdusundur|\bmoguce\b|\bvjerojatno\b|\bsumnja\b|\bmoze odgovarati\b|πιθαν|υποπτ|\bmoglich|\bverdachtig|\bfraglich|\bv\.?a\.?\b|\bwohl\b|\bвъзможно\b|\bвероятно\b|суспект|\bmogelijk\b|\bverdacht\b
```

### TEAR

```regex
\btear|\btorn\b|\brupture|\bdisruption\b|discontinuit|\bavuls|\bmacerat|\bbuckethandle\b|bucket handle|\brotura\b|\broturas\b|\bruptura|\bdesgarro|\broto\b|\bdechirure|\bdechire|\bscheur|\bruptuur|gescheurd|\briss\b|einriss|\bruptur|zerreiss|\blasion|\bausriss|\byirtik|\byirtig|\bkopma\b|butunluk kaybi|\brupturu\b|devamsizlik|\brupture\b|\bdevamliligi secilememis|\bpuknuce|\bprekid\b|\bpukotin|\bruptur|ρηξη|ρηξις|ρηγμα|ασυνεχεια|руптура|разкъсв|разрив|скъсв|\bлезия\b
```

### DEGEN

```regex
degenerat|\bmucoid\b|\bmyxoid\b|\bfray|\bfissur|dejeneratif|\bmukoid\b|degenerativn|εκφυλ|дегенерат|\bμυξοειδ|\bμυξωδ|\bmeniskopat|\bmeniscopath|\bmuco ?ide\b|aufgefasert|\bdejenerasyon\b
```

### INJURY

```regex
\binjur|\bsprain|\blesion|\blasion|\bedema\b|\boedema\b|\bodem\b|\bedem\b|\bοιδημα|\bодем|\bедем|\bstrain\b|\bhigh signal\b|\bsignal alteration\b|\bhiperintens|\bhyperintens|aumento de senal|alteracion de senal|cambio de senal|\bsignalanhebung|\bsignalalteration|verhoogd signaal|sinyal artis|αυξημενο σημα|повишен сигнал|\besguince\b|\bthicken|\bzadebljanje\b|\bverdikking\b|\bdistenzij|\blaksite\b|\blaxity\b|\bpartial\b|\bparcijaln|\bparcial|\bpartiel|\bpartiell
```

### _GRADE_RX

```regex
(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)[\s:]*(?:grade\s*)?([1-4]|iv|iii|ii|i)\b
```

### ANAT['ACL']

```regex
anterior cruciate|\bacl\b|cruzado anterior|\blca\b|croise anterieur|voorste kruisband|\bvkb\b|vorderes kreuzband|vorderen kreuzband|vordere kreuzband|on capraz|\bocb\b|anterior capraz|prednji krizni|prednjeg krizn|προσθι[οα][^ ]* χιαστ|προσθιου χιαστου|χιαστο[^ ]* συνδεσμ|\bχιαστ\w*|предна кръстна|предната кръстна|предна кръста|cruciate ligaments|ligamentos cruzados|ligaments croises|kruisbanden|kreuzbander|capraz baglar|krizn[a-z]* ligament[a-z]*|χιαστοι συνδεσμ|χιαστων συνδεσμ|кръстните връзки|кръстни връзки
```

### ANAT['MCL']

```regex
medial collateral|\bmcl\b|tibial collateral|colateral medial|colateral interno|\blcm\b|collateral medial|collateral interne|mediale collaterale|binnenband|\b(mediale|laterale) banden\b|\bcollaterale banden\b|innenband|mediales? kollateral|\bic yan bag|medial kollateral|\biyb\b|medyal kollateral|medijalni kolateraln|medijalnog kolateraln|εσω πλαγι|εσωτερικο πλαγι|\bπλαγι\w* συνδεσμ|\bπλαγιοι\b|медиален колатерал|вътрешна странична|\bколатерал\w*|\bcolaterales\b|\bcollateraux\b|\bcollateralen\b|\bkolateralni\b|collateral ligaments|ligamentos colaterales|ligaments collateraux|collaterale banden|kollateralbander|seitenbander|yan baglar|kolateraln[a-z]* ligament[a-z]*|πλαγιοι συνδεσμ|πλαγιων συνδεσμ|колатерални връзки|страничните връзки
```

### ANAT['Medial Meniscus']

```regex
medial meniscus|\bmm\b(?= tear)|medial menisc|menisco medial|menisco interno|menisque medial|menisque interne|mediale meniscus|binnenmeniscus|innenmeniskus|medialen? meniskus|innenmeniskushinterhorn|medyal menisk|\bic menisk|medijalni meniskus|medijalnog meniskusa|medijalnom meniskusu|medijaln\w* menisk\w*|\bmedijalnog meniska\b|medijalni menisk|εσω μηνισκ|μηνισκ[^ ]* του εσω|εσω διαμερισμα[^.]{0,40}μηνισκ|медиалния менискус|медиален менискус|вътрешния менискус|oba meniska|both menisci|ambos meniscos|beide menisci|her iki menisku|amfoteroi\w* mhnisk|αμφοτερ\w* μηνισκ|двата менискуса|medial (and|&) lateral menisc
```

### ANAT['Lateral Meniscus']

```regex
lateral meniscus|lateral menisc|menisco lateral|menisco externo|menisque lateral|menisque externe|laterale meniscus|buitenmeniscus|aussenmeniskus|lateralen? meniskus|aussenmeniskushinterhorn|lateral menisk|\bdis menisk|lateralni meniskus|lateralnog meniskusa|lateralnom meniskusu|lateraln\w* menisk\w*|\blateralnog meniska\b|εξω μηνισκ|μηνισκ[^ ]* του εξω|εξω διαμερισμα[^.]{0,40}μηνισκ|латералния менискус|латерален менискус|външния менискус|oba meniska|both menisci|ambos meniscos|beide menisci|her iki menisku|αμφοτερ\w* μηνισκ|двата менискуса|medial (and|&) lateral menisc
```

### OA_EVIDENCE

```regex
osteoarthrit|\barthros|\bgonarthros|\bosteoarthros|chondropath|chondromalac|condropat|condromalac|\bchondros|\bchondrosis\b|chondral (loss|defect|ulcer|thinning|injury|fissur|wear)|cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)|(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage|articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)|osteophyt|osteofit|osteofyt|osteofito|osteophyten|spurring|joint space narrowing|pinzamiento articular|reduced joint space|kikirdak kayb|kikirdak incelme|kondropati|kondral|kikirdak dejener|eklem aralig\w* daral|eklem mesafesi daral|kikirdak kalinlig\w* azal|kraakbeen|gonartrose|artrose|\bknorpel|arthrose|gonarthrose|hrskavic|hondromalac|artroz|osteoartrit|artrotsk|artrotick|\boa promjen|\boa\b|degenerativne promjene hrskav|χονδρ[^ ]*παθ|αρθριτ|αρθρωσ|οστεοφυτ|χονδρομαλακ|αρθρικου χονδρου|εξαλειψη του αρθρικου χονδρου|διαβρωση του αρθρικου χονδρ|λεπτυνση[^.]{0,30}χονδρ|φθορα[^.]{0,20}χονδρ|артроз|хондропат|остеофит|хрущял[^.]{0,40}(изтън|увред|дефект|липс)|изтъняване[^.]{0,30}хрущял|хондромалац|ulcera[s]? condral|cartilago[^.]{0,25}(perdida|adelgaz)|icrs grade|icrs\b|outerbridge|\bdenudation\b|denudacij|erozivne promjene|\berosion of[^.]{0,20}cartilage|kraakbeenlijden|kraakbeenverlies
```

### TF_SITE

```regex
compartment|compartimento|compartiment|kompartman|kompartiment|kompartment|odjelj|διαμερισμα|компартм|\bотдел|femorotibial|tibiofemoral|femoro tibial|femorotibiaal|femorotibijaln|феморотибиал|\bft zglob|tibiofemoraln|condyle|condilo|kondyl|kondil|condyl|κονδυλ|кондил|\bplateau|\bplato\b|platillo|meseta|плато|tibiaplateau|tibijaln\w* plato|tibyal plato|tibia plato|κνημιαι|μηριαι|weightbearing|weightbaring|zona de carga|dragende deel|agirlik tasiyan|\bfemur\b|\btibia\b|\bfemoral\b|\btibial\b|\bfemura\b|\btibije\b|\bmesarthrio\b|μεσαρθριο
```

### PF_SITE

```regex
patellofemoral|femoropatellar|femoropatelar|patelofemoral|retropatellar|retrorotulian|trochlea|troclea|troklea|trochlear|trohlej|τροχιλ|\bpatella|\bpatellar|rotulian|\brotula\b|\bpatele\b|patellofemoraal|femoropatellair|επιγονατιδ|μηροεπιγονατιδ|пател|феморопател|anterior compartment|compartimento anterior|prednj\w* odjeljk|\bfp zglob|\bpf zglob|\bfaset|\bfacet|patellofemoraln
```

### SIDE_MEDIAL

```regex
\bmedial\w*|\bmedyal\w*|\bmedijaln\w*|\bmediaal\w*|\bmediale\w*|\binterno\b|\binterna\b|\binternos\b|\binterne\b|\binnen\w*|\bic\b|\bunutarnj\w*|\bεσω\w*|\bεσωτερικ\w*|\bмедиал\w*|\bвътреш\w*|\bbinnen\w*|\bmediaal\b|\bmediales?\b
```

### SIDE_LATERAL

```regex
\blateral\w*|\bexterno\b|\bexterna\b|\bexternos\b|\bexterne\b|\bdis\b|\blateraln\w*|\baussen\w*|\bbuiten\w*|\bεξω\w*|\bεξωτερικ\w*|\bлатерал\w*|\bвъншн\w*|\bvanjsk\w*
```

### SIDE_ANTERIOR

```regex
\banterior\w*|\bant\b|\bon\b|\bprednj\w*|\bvorder\w*|\bvoorste\b|\bπροσθι\w*|\bпредн\w*|\banteriyor\w*|\bavant\b|\banterieur\w*
```

### GLOBAL_OA

```regex
tri ?compartment|all three compartment|global(ised)? (oa|osteoarthrit)|\bgonarthros|\bgonartros|\bgonarthrose|\bgonartrose|gonartro|goanrtrot|gonartrot|osteoarthritis of the knee|artrosis (de |)(la )?rodilla|knee osteoarthrit|\bdiz osteoartrit|\bgonartroz|artroza koljena|οστεοαρθριτιδα|αρθριτιδα του γονατος|εκφυλιστικη οστεοαρθριτ|артроза на колянната|гонартроз|degenerative joint disease|\bdjd\b|three compartments|compartmens|compartments
```

### DIRECT['Effusion']

```regex
\beffusion|joint fluid|intra ?articular fluid|\bhydrops\b|\bhemarthros|\bhaemarthros|derrame articular|\bderrame\b|liquido articular|hemartrosis|epanchement|gewrichtsvocht|\bvocht\b|gewrichtseffusie|opzetting van suprapatell|gelenkerguss|\berguss\b|gelenksergu|gelenksflussigkeit|eklem\w* ic\w* sivi|efuzyon|eklem sivisi|eklem mesafesinde sivi|sivi (miktari|artisi|birikimi)|sivi artis|\bsivi\b[^.]{0,25}artmis|\bizljev|\bizliv|zglobn[^ ]* tekucin|\bhidrops\b|αρθρικ[^ ]* υγρ|υγρου ενδαρθρικα|ενδαρθρικ[^ ]* υγρ|ποσοτητα υγρου|ενδαρθρικ|αρθρικη συλλογη|υγρο στην αρθρωση|υγρου στην αρθρωση|συλλογη υγρου|ενθαρθρικ|ставен излив|излив|ставна течност|синовиална течност
```

### DIRECT['Synovitis']

```regex
synovit|sinovit|synovial (thickening|proliferation|hypertroph)|thicken\w* synovial|hypertroph\w* of the synovium|synoviale? (verdikking|proliferatie)|verdikkingen van (het )?synovium|synovialitis|synovialis(verdickung|proliferation)|reizsynovial|sinovijalitis|sinovitis|zadebljanje sinovij|proliferacij\w* sinovij|sinovijaln\w* proliferacij|υμενιτιδα|συνοβιτιδα|υμενικ[^ ]* υπερτροφ|αρθρικου υμεν|παχυνση[^.]{0,20}υμεν|υμενα|синовит|синовиал[^ ]* (задебел|пролифер)|\bpannus\b|\bhoffit|sinovyal\w* (kalinlas|proliferas)|sinovyal hipertrof|\bartrit\b|\barthritis\b
```

### DIRECT["Baker's"]

```regex
baker|popliteal cyst|quiste popliteo|quistes popliteos|kyste poplite|popliteale? cyst|poplitealzyste|bakerzyste|popliteal kist|\bbakerova\b|poplitealn[^ ]* cist|popliteal\w* cist|κυστη baker|πολυχωρη συνοβιακη κυστη|κυστη του baker|συνοβιακη κυστη|κυστη τυπου baker|киста на бейкър|бейкърова киста|поплитеална киста|бекеров|gastrocnemio ?semimembranos|gastrocnemius semimembranosus burs
```

### DIRECT['Contusion']

```regex
\bcontusion|bone bruise|bone marrow (o?edema|contusion)|marrow o?edema|\bkontuz|medular bone o?edema|osseous contusion|contusion osea|edema oseo|edema de medula osea|contusiones oseas|oedeme osseux|contusion osseuse|botcontusie|botoedeem|beenmergoedeem|botmergoedeem|knochenmarkodem|knochenodem|knochenmarksodem|kontusion|kemik kontuzyonu|kemik iligi odemi|kemik odemi|kemik iliginde odem|kontuzyonel kemik|kemik iligi odemleri|kostani edem|edem kosti|kontuzij|kostane srzi[^.]{0,20}edem|οστεομυελικ[^ ]* οιδημα|οστικο οιδημα|μυελικο οιδημα|οστικο μωλωπ|костномозъчен едем|костен едем|контузионен|костно мозъчен едем
```

### DIRECT['Fracture']

```regex
\bfractur|\bfract\b|\bfractura|\bfracturas\b|\bfractuur|\bbreuk\b|\bfraktur|\bbruch\b|\bkirik\b|\bkirigi\b|\bkiri[kg]\w*|\bprijelom|impresijsk[^ ]* fraktur|impaktcij|καταγμα|καταγματ|фрактур|счупван|фисур|insufficiency fracture|stress fracture|avulsion fracture|subchondral fracture|subkondral kiri|impaction (fracture|injury)|osteochondral (fracture|impaction)|\bsegond\b|impactiefractuur|subchondrale impression|subchondraler? impress
```

### DECOY['Fracture']

```regex
microfractur|\bfracture (risk|prophyla)
```

### DECOY["Baker's"]

```regex
meniscal cyst|quiste meniscal|parameniscal
```

### PLURAL_MENISCI

```regex
\bmenisci\b|\bmeniscos\b|\bmenisques\b|\bmenisken\b|\bmeniskusi\b|\bmenisk\w*ler\b|\bμηνισκοι\b|\bμηνισκων\b|\bменискуси\b|\bменискусите\b|\bmenisci\w*\b
```

### ANY_SIDE

```regex
\bmedial\w*|\bmedyal\w*|\bmedijaln\w*|\bmediaal\w*|\bmediale\w*|\binterno\b|\binterna\b|\binternos\b|\binterne\b|\binnen\w*|\bic\b|\bunutarnj\w*|\bεσω\w*|\bεσωτερικ\w*|\bмедиал\w*|\bвътреш\w*|\bbinnen\w*|\bmediaal\b|\bmediales?\b|\blateral\w*|\bexterno\b|\bexterna\b|\bexternos\b|\bexterne\b|\bdis\b|\blateraln\w*|\baussen\w*|\bbuiten\w*|\bεξω\w*|\bεξωτερικ\w*|\bлатерал\w*|\bвъншн\w*|\bvanjsk\w*
```

### STEM_MENISCUS

```regex
menisc\w*|menisk\w*|μηνισκ\w*|мениск\w*
```

### STEM_CRUCIATE

```regex
cruciate|cruzado|croise|kruisband|kreuzband|capraz bag\w*|krizn\w*|χιαστ\w*|кръстн\w*|\bacl\b|\blca\b|\bvkb\b|\bocb\b|\bacb\b
```

### STEM_COLLATERAL

```regex
collateral\w*|colateral\w*|kollateral\w*|collaterale\w*|kolateraln\w*|yan bag\w*|πλαγι\w*|колатерал\w*|странич\w*|innenband\w*|binnenband\w*|\bmcl\b|\blcm\b|\biyb\b
```

### STEM_FRACTURE

```regex
fractur\w*|fraktur\w*|fractuur\w*|\bfract\b|kiri[kgğ]\w*|prijelom\w*|lom kosti|\bbreuk\w*|\bbruch\w*|καταγμα\w*|καταγματ\w*|фрактур\w*|счупван\w*|fisur\w* (osea|oseas|kost)|fissur\w* kost
```

### SEV_LOW

```regex
\bsmall\b|\bminimal\b|\btrace\b|\bmild\b|\bslight\b|\btiny\b|\bscant\b|\bdiscrete\b|\blow ?grade\b|\bincipient\b|\bleve\b|\bminim|\bpeque|\bfina\b|\bfino\b|\bligero\b|\bescaso\b|\bdiscreto\b|\bhafif\b|\baz miktarda\b|\bsilik\b|\bmanj\w*|\bblago\b|\bdiskretn|\bmalo\b|\bpocetn|\bgering|\bdiskret|\bkleine?r?\b|\bwenig\b|\bzarte?\b|\bbeperkte?\b|\bgeringe\b|\bweinig\b|\blichte?\b|\blicht\b|\bηπι|\bμικρ|\bελαχιστ|\bαρχομεν|\bминимал|\bлек|\bмалк|\bнеголям
```

### SEV_HIGH

```regex
\blarge\b|\bmarked\b|\bmassive\b|\bsevere\b|\bextensive\b|\bmoderate\b|\bgross\b|\bsignificant\b|\babundant\b|\btense\b|\bcomplete\b|\bfull ?thickness\b|\bhigh ?grade\b|\badvanced\b|\bmoderad|\bimportante\b|\bsevera?\b|\bmarcad|\bcuantios|\bespesor total\b|\bcompleta?\b|\bbelirgin\b|\byaygin\b|\bileri\b|\bciddi\b|\bbol\b|\bkomplet|\bopsezan\b|\bveliki\b|\bizrazit|\bznacajn|\bumjeren|\buznapredoval|\bpotpun|\bkompleksn|\bausgepragt|\bdeutlich|\bmassiv|\bmassig|\bgross|\buitgebreid|\bgevorderd|\bveel\b|\bmatige?\b|\bvolledig|\bμετρι|\bμεγαλ|\bεκτεταμεν|\bευμεγεθ|\bσοβαρ|\bπληρη|\bголям|\bизразен|\bзначим|\bумерен|\bобилен|\bпълн
```

### GRADE_HIGH

```regex
grade?[ao]?\s*(3|4|iii|iv)\b|icrs grade (iii|iv|3|4)|stupnja iv|stupnja iii|\bgrado (3|4)\b|\bgrad (3|4)\b|\bgrade (3|4)\b
```

### DEGENERATIVE_MARROW

```regex
subchondral|subcondral|subkondral|supkondraln|subchondraln|υποχονδρι|υπαρθρικ|субхондрал|subchondrale?|subartikuler|\bcyst|\bquist|\bzyste\b|\bcistic|reactive|reactivo|degenerative|degenerativ|reaktiv|\bcisti\b
```

### TRAUMA

```regex
\bbruise\b|\bcontusion|\bkontuz|\btrauma|\bimpaction\b|\bpivot shift\b|\bkissing\b|\bacute\b|\bagudo\b|\bakut|\bpivot kaymasi\b|\bcontusion osseuse\b|\bbone bruise\b|\bbotcontusie\b|\bконтузион|\bμωλωπ|\bkontuzij|\bimpaktcij|\bimpakcij|\bfall\b|\binjury\b|\bimpression\b
```

### SYNOVIAL_PROXY

```regex
bursit|burzit|\bbursa\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)|suprapatellar (bursitis|effusion|recess)|suprapatellar bursa|suprapatellar bursada|suprapatelarno|suprapatellaire recessus|hoffa|hoffit|plica|plika|πλικα|fat pad[^.]{0,20}(edema|oedema)|kapsul|capsul|καψ|капсул|\bpannus\b|\bsinov|\bsynov
```

## Additional inline regex patterns

Cell 17, `re.sub`:

```regex
[_\-/\\]+
```

Cell 17, `re.sub`:

```regex
[ \t]+
```

Cell 17, `re.search`:

```regex
[.;:!?>*•]$
```

Cell 17, `re.search`:

```regex
\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|ωστοσο|αλλα|но)\b
```

