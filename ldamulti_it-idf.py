import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import re
import unicodedata

class SupervisedTopicLabeler:
    """
    Hybrid topic labeler that learns from:
    1. Manually labeled data (high quality)
    2. Keyword-based labels with high confidence (semi-supervised)
    Falls back to keyword matching for low-confidence predictions.
    """
    
    def __init__(self):
        self.topic_labels = {
            0: 'Economic Development',
            1: 'Natural Resources & Energy',
            2: 'War & Conflict',
            3: 'Social Services',
            4: 'Politics & Governance',
            5: 'Art, Technology and Sport'
        }
        
        # ML model pipeline
        self.model = None
        self.trained = False
        self.metrics = {}
        
        self.multilingual_keywords = {
            'Economic Development': {
                'en': ['economy', 'economic', 'trade', 'trading', 'investment', 'invest', 'investor',
                'business', 'market', 'finance', 'financial', 'bank', 'banking',
                'growth', 'gdp', 'currency', 'forex', 'stock', 'bonds', 'capital',
                'entrepreneur', 'commerce', 'export', 'import', 'revenue', 'profit',
                'fiscal', 'monetary', 'inflation', 'debt', 'loan', 'credit', 'development',
                'corporate', 'sector', 'commodity', 'portfolio', 'startup', 'venture', 'aid'],
                'fr': ['économie', 'économique', 'commerce', 'investissement', 'investir', 'investisseur',
                'affaires', 'marché', 'finance', 'financier', 'banque', 'bancaire',
                'croissance', 'pib', 'devise', 'bourse', 'obligations', 'capital',
                'entrepreneur', 'exportation', 'importation', 'revenu', 'profit',
                'fiscal', 'monétaire', 'inflation', 'dette', 'prêt', 'crédit', 'développement',
                'entreprise', 'secteur', 'marchandise', 'portefeuille', 'startup', 'aide'],
                'ar': ['اقتصاد', 'اقتصادي', 'تجارة', 'استثمار', 'أعمال', 'سوق', 'مالية', 'بنك',
                'نمو', 'عملة', 'رأسمال', 'ربح', 'دخل', 'تصدير', 'استيراد',
                'ديون', 'قرض', 'ائتمان', 'تضخم', 'قطاع', 'شركة', 'محفظة', 'تطوير'],
                'zh-cn': ['经济', '经济的', '贸易', '投资', '商业', '市场', '金融', '银行', '增长', 'gdp', '货币', '外汇', '股票', '债券', '资本', '企业', '出口', '进口', '收入', '利润', '通货膨胀', '债务', '贷款', '信贷', '发展', '行业', '商品', '创业', '初创', '风投', '援助'],
                'ko': ['경제', '무역', '투자', '사업', '시장', '금융', '은행', '성장', 'gdp', '통화', '외환', '주식', '채권', '자본', '기업', '수출', '수입', '수익', '이익', '인플레이션', '부채', '대출', '신용', '개발', '산업', '상품', '창업', '스타트업', '벤처', '원조'],
                'es': ['economía', 'económico', 'comercio', 'inversión', 'invertir', 'inversor', 'negocios', 'mercado', 'finanzas', 'financiero', 'banco', 'banca', 'crecimiento', 'pib', 'moneda', 'divisa', 'bolsa', 'bonos', 'capital', 'emprendedor', 'exportación', 'importación', 'ingresos', 'beneficio', 'fiscal', 'monetario', 'inflación', 'deuda', 'préstamo', 'crédito', 'desarrollo', 'empresa', 'sector', 'mercancía', 'cartera', 'startup', 'ayuda'],
                'pt': ['economia', 'econômico', 'comércio', 'investimento', 'investir', 'investidor', 'negócios', 'mercado', 'finanças', 'financeiro', 'banco', 'bancário', 'crescimento', 'pib', 'moeda', 'câmbio', 'bolsa', 'títulos', 'capital', 'empreendedor', 'exportação', 'importação', 'receita', 'lucro', 'fiscal', 'monetário', 'inflação', 'dívida', 'empréstimo', 'crédito', 'desenvolvimento', 'empresa', 'setor', 'mercadoria', 'carteira', 'startup', 'ajuda'],
            },
            'Natural Resources & Energy': {
                'en': ['oil', 'petroleum', 'crude', 'mining', 'mineral', 'coal', 'gold',
                'diamond', 'energy', 'power', 'electricity', 'gas', 'natural gas',
                'renewable', 'solar', 'wind', 'hydro', 'nuclear', 'extraction',
                'resources', 'fossil', 'lithium', 'copper', 'cobalt', 'uranium',
                'refinery', 'drilling', 'pipeline', 'reserves', 'exploration',
                'hydroelectric', 'fuel', 'ore', 'geothermal', 'biomass', 'turbine', 'grid'],
                'fr': ['pétrole', 'brut', 'exploitation minière', 'mine', 'minéral', 'charbon', 'or',
                'diamant', 'énergie', 'électricité', 'gaz', 'gaz naturel',
                'renouvelable', 'solaire', 'vent', 'éolien', 'hydro', 'hydroélectrique', 'nucléaire', 'extraction',
                'ressources', 'fossile', 'lithium', 'cuivre', 'cobalt', 'uranium',
                'raffinerie', 'forage', 'pipeline', 'réserves', 'exploration',
                'carburant', 'minerai', 'géothermique', 'biomasse', 'turbine', 'réseau'],
                'ar': ['نفط', 'زيت', 'تعدين', 'معدن', 'فحم', 'ذهب',
                'ألماس', 'طاقة', 'كهرباء', 'غاز',
                'متجددة', 'شمسي', 'رياح', 'مائي', 'نووي', 'استخراج',
                'موارد', 'وقود', 'ليثيوم', 'نحاس', 'يورانيوم',
                'تكرير', 'حفر', 'خط', 'احتياطيات', 'استكشاف',
                'خامات', 'حراري', 'حيوي', 'توربين', 'شبكة'],
                'zh-cn': ['石油', '原油', '采矿', '矿产', '煤炭', '黄金', '钻石', '能源', '电力', '天然气', '可再生', '太阳能', '风能', '水电', '核能', '开采', '资源', '化石', '锂', '铜', '钴', '铀', '炼油厂', '钻探', '管道', '储备', '勘探', '燃料', '矿石', '地热', '生物质', '涡轮机', '电网'],
                'ko': ['석유', '원유', '채굴', '광물', '석탄', '금', '다이아몬드', '에너지', '전력', '천연가스', '재생에너지', '태양광', '풍력', '수력', '원자력', '채취', '자원', '화석', '리튬', '구리', '코발트', '우라늄', '정유', '시추', '파이프라인', '매장량', '탐사', '연료', '광석', '지열', '바이오매스', '터빈', '전력망'],
                'es': ['petróleo', 'crudo', 'minería', 'mineral', 'carbón', 'oro', 'diamante', 'energía', 'electricidad', 'gas', 'gas natural', 'renovable', 'solar', 'eólica', 'hidroeléctrica', 'nuclear', 'extracción', 'recursos', 'fósil', 'litio', 'cobre', 'cobalto', 'uranio', 'refinería', 'perforación', 'oleoducto', 'reservas', 'exploración', 'combustible', 'mena', 'geotérmica', 'biomasa', 'turbina', 'red'],
                'pt': ['petróleo', 'bruto', 'mineração', 'mineral', 'carvão', 'ouro', 'diamante', 'energia', 'eletricidade', 'gás', 'gás natural', 'renovável', 'solar', 'eólica', 'hidrelétrica', 'nuclear', 'extração', 'recursos', 'fóssil', 'lítio', 'cobre', 'cobalto', 'urânio', 'refinaria', 'perfuração', 'oleoduto', 'reservas', 'exploração', 'combustível', 'minério', 'geotérmica', 'biomassa', 'turbina', 'rede'],
            },
            'War & Conflict': {
                'en': ['war', 'conflict', 'violence', 'military', 'army', 'soldier', 'troop',
                'rebel', 'insurgent', 'militant', 'attack', 'bombing', 'strike',
                'terrorism', 'terrorist', 'extremist', 'jihadist', 'boko haram',
                'al-shabaab', 'militia', 'armed group', 'peacekeeping', 'ceasefire',
                'casualties', 'killed', 'wounded', 'battle', 'fighting', 'clash',
                'combat', 'offensive', 'raid', 'ambush', 'siege', 'refugee',
                'displacement', 'humanitarian crisis', 'genocide', 'ethnic cleansing',
                'civil war', 'coup', 'rebellion', 'uprising', 'unrest', 'protest violence',
                'security forces', 'defense', 'weapon', 'arms', 'ammunition',
                'warfare', 'death', 'kill', 'guns', 'gun', 'bomb', 'dead', 'bodies', 'fire',
                'fight', 'violent', 'assault', 'airstrike', 'shelling', 'hostilities'],
                'fr': ['guerre', 'conflit', 'violence', 'militaire', 'armée', 'soldat', 'troupe',
                'rebelle', 'insurgé', 'militant', 'attaque', 'bombardement', 'frappe',
                'terrorisme', 'terroriste', 'extrémiste', 'milice', 'cessez-le-feu',
                'victimes', 'tués', 'tué', 'blessés', 'bataille', 'combat', 'combats',
                'offensive', 'raid', 'siège', 'réfugié',
                'crise humanitaire', 'génocide',
                'guerre civile', 'coup', 'rébellion', 'soulèvement',
                'forces de sécurité', 'défense', 'arme', 'armes', 'munitions',
                'mort', 'tuer', 'fusil', 'bombe', 'morts', 'corps', 'feu',
                'violent', 'assaut', 'frappe aérienne', 'pilonnage', 'hostilités'],
                'ar': ['حرب', 'صراع', 'نزاع', 'عنف', 'عسكري', 'جيش', 'جندي', 'قوات',
                'متمرد', 'متشدد', 'هجوم', 'قصف', 'ضربة',
                'إرهاب', 'إرهابي', 'متطرف', 'ميليشيا', 'هدنة',
                'ضحايا', 'قتلى', 'مقتول', 'جرحى', 'معركة', 'قتال',
                'هجوم', 'حصار', 'لاجئ',
                'أزمة إنسانية', 'إبادة',
                'حرب أهلية', 'انقلاب', 'ثورة', 'تمرد',
                'قوات الأمن', 'دفاع', 'سلاح', 'أسلحة', 'ذخيرة',
                'موت', 'قتل', 'بندقية', 'قنبلة', 'جثة', 'جثث', 'نار',
                'عدوان', 'مسلح', 'عداوة']
                ,
                'zh-cn': ['战争', '冲突', '暴力', '军事', '军队', '士兵', '叛军', '武装分子', '攻击', '爆炸', '空袭', '恐怖主义', '恐怖分子', '极端分子', '民兵', '休战', '伤亡', '死亡', '受伤', '战斗', '交战', '作战', '进攻', '突袭', '伏击', '围困', '难民', '流离失所', '人道危机', '种族灭绝', '内战', '政变', '叛乱', '起义', '动乱', '抗议', '安全部队', '防御', '武器', '弹药', '枪支', '炸弹', '敌对'],
                'ko': ['전쟁', '분쟁', '폭력', '군사', '군대', '병사', '반군', '무장단체', '공격', '폭탄', '공습', '테러', '테러리스트', '극단주의자', '민병대', '휴전', '사상자', '사망', '부상', '전투', '교전', '작전', '공세', '급습', '매복', '포위', '난민', '인도적 위기', '집단학살', '내전', '쿠데타', '반란', '봉기', '소요', '시위', '치안군', '방어', '무기', '탄약', '총기', '폭발물', '적대'],
                'es': ['guerra', 'conflicto', 'violencia', 'militar', 'ejército', 'soldado', 'rebelde', 'insurgente', 'militante', 'ataque', 'bombardeo', 'ofensiva', 'terrorismo', 'terrorista', 'extremista', 'milicia', 'alto el fuego', 'víctimas', 'muertos', 'heridos', 'batalla', 'combates', 'choque', 'combate', 'incursión', 'emboscada', 'asedio', 'refugiado', 'desplazamiento', 'crisis humanitaria', 'genocidio', 'guerra civil', 'golpe', 'rebelión', 'levantamiento', 'disturbios', 'protesta', 'fuerzas de seguridad', 'defensa', 'arma', 'armas', 'munición'],
                'pt': ['guerra', 'conflito', 'violência', 'militar', 'exército', 'soldado', 'rebelde', 'insurgente', 'militante', 'ataque', 'bombardeio', 'ofensiva', 'terrorismo', 'terrorista', 'extremista', 'milícia', 'cessar-fogo', 'vítimas', 'mortos', 'feridos', 'batalha', 'combates', 'confronto', 'operação', 'incursão', 'emboscada', 'cerco', 'refugiado', 'deslocamento', 'crise humanitária', 'genocídio', 'guerra civil', 'golpe', 'rebelião', 'levante', 'distúrbios', 'protesto', 'forças de segurança', 'defesa', 'arma', 'armas', 'munição']
            },
            'Social Services': {
                'en': ['health', 'healthcare', 'hospital', 'medical', 'doctor', 'nurse', 'physician',
                'clinic', 'patient', 'disease', 'vaccine', 'medicine', 'treatment', 'therapy',
                'education', 'school', 'university', 'teacher', 'student', 'learning',
                'training', 'literacy', 'scholarship', 'welfare', 'social',
                'pandemic', 'epidemic', 'curriculum', 'tuition', 'degree','covid','schools',
                'virus', 'malaria', 'tuberculosis', 'hiv', 'aids', 'immunization',
                'maternal', 'childcare', 'hunger', 'nutrition', 'water', 'sanitation', 'shelter'],
                'fr': ['santé', 'soins', 'hôpital', 'médical', 'médecin', 'infirmier', 'infirmière',
                'clinique', 'patient', 'maladie', 'vaccin', 'médicament', 'traitement', 'thérapie',
                'éducation', 'école', 'université', 'enseignant', 'professeur', 'étudiant', 'apprentissage',
                'formation', 'alphabétisation', 'bourse', 'bien-être', 'social',
                'pandémie', 'épidémie', 'programme', 'diplôme', 'covid',
                'virus', 'paludisme', 'tuberculose', 'vih', 'sida', 'immunisation',
                'maternelle', 'garde', 'faim', 'nutrition', 'eau', 'assainissement', 'abri'],
                'ar': ['صحة', 'رعاية', 'مستشفى', 'طبي', 'طبيب', 'ممرضة',
                'عيادة', 'مريض', 'مرض', 'لقاح', 'دواء', 'علاج',
                'تعليم', 'مدرسة', 'جامعة', 'معلم', 'طالب', 'تعلم',
                'تدريب', 'محو الأمية', 'منحة', 'رعاية', 'اجتماعي',
                'جائحة', 'وباء', 'منهج', 'درجة', 'كوفيد',
                'فيروس', 'ملاريا', 'سل', 'إيدز', 'تحصين',
                'أمومة', 'جوع', 'تغذية', 'ماء', 'صرف', 'ملجأ']
                ,
                'zh-cn': ['健康', '医疗', '医院', '医生', '护士', '诊所', '患者', '疾病', '疫苗', '药物', '治疗', '疗法', '教育', '学校', '大学', '教师', '学生', '学习', '培训', '识字', '奖学金', '福利', '社会', '大流行', '流行病', '课程', '学费', '学位', '新冠', '病毒', '疟疾', '结核病', '艾滋病', '免疫接种', '母婴', '儿童护理', '饥饿', '营养', '水', '卫生', '住所'],
                'ko': ['건강', '의료', '병원', '의사', '간호사', '클리닉', '환자', '질병', '백신', '약', '치료', '요법', '교육', '학교', '대학', '교사', '학생', '학습', '훈련', '문해', '장학금', '복지', '사회', '팬데믹', '전염병', '교육과정', '등록금', '학위', '코로나', '바이러스', '말라리아', '결핵', 'HIV', '에이즈', '예방접종', '산모', '보육', '굶주림', '영양', '물', '위생', '주거'],
                'es': ['salud', 'atención', 'hospital', 'médico', 'doctor', 'enfermera', 'clínica', 'paciente', 'enfermedad', 'vacuna', 'medicamento', 'tratamiento', 'terapia', 'educación', 'escuela', 'universidad', 'maestro', 'profesor', 'estudiante', 'aprendizaje', 'formación', 'alfabetización', 'beca', 'bienestar', 'social', 'pandemia', 'epidemia', 'currículo', 'matrícula', 'título', 'covid', 'virus', 'malaria', 'tuberculosis', 'vih', 'sida', 'inmunización', 'materno', 'guardería', 'hambre', 'nutrición', 'agua', 'saneamiento', 'refugio'],
                'pt': ['saúde', 'cuidados', 'hospital', 'médico', 'doutor', 'enfermeira', 'clínica', 'paciente', 'doença', 'vacina', 'medicamento', 'tratamento', 'terapia', 'educação', 'escola', 'universidade', 'professor', 'estudante', 'aprendizagem', 'formação', 'alfabetização', 'bolsa', 'bem-estar', 'social', 'pandemia', 'epidemia', 'currículo', 'propina', 'diploma', 'covid', 'vírus', 'malária', 'tuberculose', 'vih', 'sida', 'imunização', 'materno', 'creche', 'fome', 'nutrição', 'água', 'saneamento', 'abrigo']
            },
            
            'Politics & Governance': {
                'en': ['politics', 'political', 'government', 'governance', 'president',
                'minister', 'parliament', 'congress', 'election', 'vote', 'voting',
                'democracy', 'policy', 'law', 'legislation', 'regulation', 'cabinet',
                'opposition', 'party', 'campaign', 'referendum', 'constitution',
                'diplomacy', 'treaty', 'summit', 'senator', 'governor', 'mayor','administration',
                'prime minister', 'reform', 'ruling', 'rebellion', 'coup',
                'senate', 'judiciary', 'ballot', 'diplomat', 'sanctions', 'sovereignty'],
                'fr': ['politique', 'gouvernement', 'gouvernance', 'président',
                'ministre', 'parlement', 'congrès', 'élection', 'vote', 'scrutin',
                'démocratie', 'politique', 'loi', 'législation', 'règlement', 'cabinet',
                'opposition', 'parti', 'campagne', 'référendum', 'constitution',
                'diplomatie', 'traité', 'sommet', 'sénateur', 'gouverneur', 'maire', 'administration',
                'premier ministre', 'réforme', 'règne', 'rébellion', 'coup',
                'sénat', 'judiciaire', 'diplomate', 'sanctions', 'souveraineté'],
                'ar': ['سياسة', 'سياسي', 'حكومة', 'حكم', 'رئيس',
                'وزير', 'برلمان', 'مجلس', 'انتخابات', 'تصويت',
                'ديمقراطية', 'قانون', 'تشريع', 'قانون', 'وزراء',
                'معارضة', 'حزب', 'حملة', 'استفتاء', 'دستور',
                'دبلوماسية', 'معاهدة', 'قمة', 'حاكم', 'عمدة', 'إدارة',
                'وزير أول', 'إصلاح', 'حكم', 'ثورة', 'انقلاب',
                'قضائي', 'دبلوماسي', 'عقوبات', 'سيادة'],
                'zh-cn': ['政治', '政府', '治理', '总统', '部长', '议会', '国会', '选举', '投票', '民主', '政策', '法律', '立法', '监管', '内阁', '反对派', '政党', '竞选', '公投', '宪法', '外交', '条约', '峰会', '参议员', '州长', '市长', '行政', '总理', '改革', '执政', '叛乱', '政变', '参议院', '司法', '选票', '外交官', '制裁', '主权'],
                'ko': ['정치', '정부', '거버넌스', '대통령', '장관', '의회', '국회', '선거', '투표', '민주주의', '정책', '법률', '입법', '규제', '내각', '야당', '정당', '선거운동', '국민투표', '헌법', '외교', '조약', '정상회담', '상원의원', '주지사', '시장', '행정부', '총리', '개혁', '집권', '반란', '쿠데타', '상원', '사법부', '투표용지', '외교관', '제재', '주권'],
                'es': ['política', 'gobierno', 'gobernanza', 'presidente', 'ministro', 'parlamento', 'congreso', 'elección', 'voto', 'votar', 'democracia', 'política pública', 'ley', 'legislación', 'regulación', 'gabinete', 'oposición', 'partido', 'campaña', 'referéndum', 'constitución', 'diplomacia', 'tratado', 'cumbre', 'senador', 'gobernador', 'alcalde', 'administración', 'primer ministro', 'reforma', 'gobernar', 'rebelión', 'golpe', 'senado', 'poder judicial', 'papeleta', 'diplomático', 'sanciones', 'soberanía'],
                'pt': ['política', 'governo', 'governança', 'presidente', 'ministro', 'parlamento', 'congresso', 'eleição', 'voto', 'democracia', 'política pública', 'lei', 'legislação', 'regulação', 'gabinete', 'oposição', 'partido', 'campanha', 'referendo', 'constituição', 'diplomacia', 'tratado', 'cúpula', 'senador', 'governador', 'prefeito', 'administração', 'primeiro-ministro', 'reforma', 'governar', 'rebelião', 'golpe', 'senado', 'judiciário', 'cédula', 'diplomata', 'sanções', 'soberania']
            },
            'Art, Technology and Sport': {
                'en': ['art', 'artist', 'music', 'musician', 'painting', 'sculpture', 'gallery', 'museum',
                'exhibition', 'performance', 'theatre', 'theater', 'film', 'cinema', 'movie',
                'sport', 'sports', 'football', 'soccer', 'basketball', 'tennis', 'athletics',
                'athlete', 'championship', 'tournament', 'league', 'match', 'game', 'player',
                'team', 'coach', 'olympic', 'olympics', 'medal', 'victory', 'champion',
                'culture', 'cultural', 'heritage', 'festival', 'dance'],
                'fr': ['art', 'artiste', 'musique', 'musicien', 'peinture', 'sculpture', 'galerie', 'musée',
                'exposition', 'spectacle', 'théâtre', 'film', 'cinéma',
                'sport', 'sports', 'football', 'basket', 'tennis', 'athlétisme',
                'athlète', 'championnat', 'tournoi', 'ligue', 'match', 'jeu', 'joueur',
                'équipe', 'entraîneur', 'olympique', 'olympiques', 'médaille', 'victoire', 'champion',
                'culture', 'culturel', 'patrimoine', 'festival', 'danse', 'compétition'],
                'ar': ['فن', 'فنان', 'موسيقى', 'موسيقي', 'رسم', 'نحت', 'معرض', 'متحف',
                'عرض', 'أداء', 'مسرح', 'فيلم', 'سينما',
                'رياضة', 'رياضي', 'كرة قدم', 'كرة سلة', 'تنس', 'ألعاب قوى',
                'رياضي', 'بطولة', 'دوري', 'مباراة', 'لعبة', 'لاعب',
                'فريق', 'مدرب', 'أولمبي', 'أولمبياد', 'ميدالية', 'فوز', 'بطل',
                'ثقافة', 'ثقافي', 'تراث', 'مهرجان', 'رقص', 'منافسة']
                ,
                'zh-cn': ['艺术', '艺术家', '音乐', '音乐家', '绘画', '雕塑', '画廊', '博物馆', '展览', '表演', '剧院', '电影', '影院', '体育', '足球', '篮球', '网球', '田径', '运动员', '锦标赛', '比赛', '联赛', '比赛', '球员', '球队', '教练', '奥运会', '奖牌', '胜利', '冠军', '文化', '文化的', '遗产', '节日', '舞蹈', '竞赛'],
                'ko': ['예술', '예술가', '음악', '음악가', '그림', '조각', '갤러리', '박물관', '전시회', '공연', '극장', '영화', '시네마', '스포츠', '축구', '농구', '테니스', '육상', '선수', '선수권', '토너먼트', '리그', '경기', '게임', '선수', '팀', '코치', '올림픽', '메달', '승리', '챔피언', '문화', '문화적', '유산', '축제', '춤', '대회'],
                'es': ['arte', 'artista', 'música', 'músico', 'pintura', 'escultura', 'galería', 'museo', 'exposición', 'actuación', 'teatro', 'cine', 'película', 'deporte', 'deportes', 'fútbol', 'baloncesto', 'tenis', 'atletismo', 'atleta', 'campeonato', 'torneo', 'liga', 'partido', 'juego', 'jugador', 'equipo', 'entrenador', 'olímpico', 'olimpiadas', 'medalla', 'victoria', 'campeón', 'cultura', 'cultural', 'patrimonio', 'festival', 'danza', 'competición'],
                'pt': ['arte', 'artista', 'música', 'músico', 'pintura', 'escultura', 'galeria', 'museu', 'exposição', 'espetáculo', 'teatro', 'cinema', 'filme', 'esporte', 'esportes', 'futebol', 'basquete', 'tênis', 'atletismo', 'atleta', 'campeonato', 'torneio', 'liga', 'partida', 'jogo', 'jogador', 'equipe', 'treinador', 'olímpico', 'olimpíadas', 'medalha', 'vitória', 'campeão', 'cultura', 'cultural', 'patrimônio', 'festival', 'dança', 'competição']
            }
        }
    
    def preprocess_text(self, text, lang='en'):
        """Clean and normalize text for feature extraction"""
        if not isinstance(text, str):
            return ''
        
        text = text.lower()
        
        # Language-specific normalization
        if lang in {'en', 'fr', 'es', 'pt'}:
            text = self._strip_accents(text)
        elif lang == 'ar':
            text = self._normalize_arabic(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _strip_accents(self, s):
        return ''.join(c for c in unicodedata.normalize('NFKD', s) 
                      if not unicodedata.combining(c))
    
    def _normalize_arabic(self, s):
        s = re.sub('[\u064B-\u0652]', '', s)  # Remove diacritics
        s = s.replace('\u0649', '\u064A')     # Normalize alef/yeh
        return s
    
    def train(self, manual_df, keyword_df=None, keyword_confidence_threshold=0.5,
              text_cols=['title', 'description'], manual_label_col='topic_label',
              keyword_label_col='predicted_label_id', keyword_conf_col='prediction_confidence'):
        """
        Train the model on BOTH manual labels AND high-confidence keyword labels
        
        Parameters:
        -----------
        manual_df : DataFrame
            Your 200 manually labeled examples with ground truth
        keyword_df : DataFrame (optional)
            Output from your previous keyword labeling code
        keyword_confidence_threshold : float
            Only use keyword labels with confidence >= this threshold (default: 0.5)
        text_cols : list
            Text columns to combine for features
        manual_label_col : str
            Column name for manual ground truth labels (0-5)
        keyword_label_col : str
            Column name for keyword-predicted labels
        keyword_conf_col : str
            Column name for keyword confidence scores
        """
        print("="*80)
        print("TRAINING SEMI-SUPERVISED MODEL")
        print("="*80)
        
        # Process manual labels
        manual_df = manual_df.copy()
        if isinstance(text_cols, list):
            manual_df['combined_text'] = manual_df[text_cols].fillna('').agg(' '.join, axis=1)
        else:
            manual_df['combined_text'] = manual_df[text_cols].fillna('')
        
        if 'language_code' in manual_df.columns:
            manual_df['processed_text'] = manual_df.apply(
                lambda row: self.preprocess_text(row['combined_text'], row['language_code']),
                axis=1
            )
        else:
            manual_df['processed_text'] = manual_df['combined_text'].apply(self.preprocess_text)
        
        # Start with manual labels
        X_manual = manual_df['processed_text']
        y_manual = manual_df[manual_label_col]
        
        # Remove NaN labels
        valid_mask = y_manual.notna()
        X_manual = X_manual[valid_mask]
        y_manual = y_manual[valid_mask]
        
        print(f"\n📊 MANUAL LABELS: {len(X_manual)} examples")
        print(f"   Topic distribution:\n{y_manual.value_counts()}\n")
        
        # Add high-confidence keyword labels if provided
        if keyword_df is not None:
            print(f"📊 KEYWORD LABELS: Processing {len(keyword_df)} examples...")
            
            keyword_df = keyword_df.copy()
            
            # Filter for high confidence AND classified (not 'Unclassified')
            high_conf_mask = (
                (keyword_df[keyword_conf_col] >= keyword_confidence_threshold) &
                (keyword_df[keyword_label_col].notna()) &
                (keyword_df[keyword_label_col] != 'Unclassified')
            )
            
            keyword_df_filtered = keyword_df[high_conf_mask].copy()
            print(f"   Filtered to {len(keyword_df_filtered)} high-confidence examples "
                  f"(threshold: {keyword_confidence_threshold})")
            
            if len(keyword_df_filtered) > 0:
                # Process keyword data
                if isinstance(text_cols, list):
                    keyword_df_filtered['combined_text'] = keyword_df_filtered[text_cols].fillna('').agg(' '.join, axis=1)
                else:
                    keyword_df_filtered['combined_text'] = keyword_df_filtered[text_cols].fillna('')
                
                if 'language_code' in keyword_df_filtered.columns:
                    keyword_df_filtered['processed_text'] = keyword_df_filtered.apply(
                        lambda row: self.preprocess_text(row['combined_text'], row['language_code']),
                        axis=1
                    )
                else:
                    keyword_df_filtered['processed_text'] = keyword_df_filtered['combined_text'].apply(self.preprocess_text)
                
                X_keyword = keyword_df_filtered['processed_text']
                y_keyword = keyword_df_filtered[keyword_label_col]
                
                print(f"   Topic distribution:\n{y_keyword.value_counts()}\n")
                
                # Combine manual + keyword labels
                X_combined = pd.concat([X_manual, X_keyword], ignore_index=True)
                y_combined = pd.concat([y_manual, y_keyword], ignore_index=True)
                
                print(f"✅ TOTAL TRAINING DATA: {len(X_combined)} examples")
                print(f"   - Manual: {len(X_manual)} ({len(X_manual)/len(X_combined)*100:.1f}%)")
                print(f"   - Keyword (high-conf): {len(X_keyword)} ({len(X_keyword)/len(X_combined)*100:.1f}%)")
            else:
                print("   ⚠️  No high-confidence keyword labels found, using manual only")
                X_combined = X_manual
                y_combined = y_manual
        else:
            print("   ℹ️  No keyword data provided, using manual labels only")
            X_combined = X_manual
            y_combined = y_manual
        
        print(f"\n📈 COMBINED TOPIC DISTRIBUTION:")
        print(y_combined.value_counts())
        print()
        
        # Create pipeline: TF-IDF + Logistic Regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),        # Unigrams and bigrams
                min_df=2,                   # Ignore rare terms
                max_df=0.8,                 # Ignore too common terms
                sublinear_tf=True           # Log scaling
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',    # Handle class imbalance
                C=1.0,
                random_state=42
            ))
        ])
        
        # Cross-validation
        print("🔄 Running 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_combined, y_combined, cv=5, scoring='accuracy')
        prec_scores = cross_val_score(self.model, X_combined, y_combined, cv=5, scoring='precision_weighted')
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"   CV Precision (weighted): {prec_scores.mean():.3f} (+/- {prec_scores.std():.3f})")
        
        # Train final model
        print("\n🎯 Training final model...")
        self.model.fit(X_combined, y_combined)
        self.trained = True
        
        # Training set performance
        y_pred = self.model.predict(X_combined)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(y_combined, y_pred, average='weighted', zero_division=0)
        train_acc = (y_combined == y_pred).mean()
        print("\n" + "="*80)
        print("TRAINING SET PERFORMANCE")
        print("="*80)
        print(classification_report(y_combined, y_pred, 
                                   target_names=list(self.topic_labels.values()),
                                   zero_division=0))
        
        # Show which manual labels the model agrees/disagrees with
        if len(X_manual) > 0:
            y_manual_pred = self.model.predict(X_manual)
            manual_accuracy = (y_manual == y_manual_pred).mean()
            print(f"\n✅ Agreement with manual labels: {manual_accuracy:.1%}")
            
            disagreements = (y_manual != y_manual_pred).sum()
            if disagreements > 0:
                print(f"   ⚠️  {disagreements} disagreements - review these for quality check")
        self.metrics = {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'cv_precision_mean': float(prec_scores.mean()),
            'cv_precision_std': float(prec_scores.std()),
            'train_accuracy': float(train_acc),
            'train_precision_weighted': float(train_prec)
        }
        
        return self
    
    def predict(self, df, text_cols=['title', 'description'], 
                confidence_threshold=0.5, use_keyword_fallback=True):
        """
        Predict topics for new articles
        
        Parameters:
        -----------
        confidence_threshold : float
                            If ML prediction confidence < threshold, use keyword fallback
        use_keyword_fallback : bool
            Whether to use keyword matching for low-confidence cases
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n" + "="*80)
        print("PREDICTING TOPICS")
        print("="*80)
        
        df = df.copy()
        
        # Combine text
        if isinstance(text_cols, list):
            df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
        else:
            df['combined_text'] = df[text_cols].fillna('')
        
        # Preprocess
        if 'language_code' in df.columns:
            df['processed_text'] = df.apply(
                lambda row: self.preprocess_text(row['combined_text'], row['language_code']),
                axis=1
            )
        else:
            df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # ML predictions
        print(f"🤖 Running ML predictions on {len(df)} articles...")
        X = df['processed_text']
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = probabilities.max(axis=1)
        
        df['predicted_label_id'] = predictions
        df['predicted_label_name'] = df['predicted_label_id'].map(self.topic_labels)
        df['prediction_confidence'] = confidences
        df['prediction_source'] = 'ml_model'  # Track prediction source
        
        # Keyword fallback for low-confidence predictions (vectorized to avoid chained assignment)
        if use_keyword_fallback:
            low_conf_mask = df['prediction_confidence'] < confidence_threshold
            n_low_conf = low_conf_mask.sum()
            
            if n_low_conf > 0:
                print(f"🔄 Applying keyword fallback to {n_low_conf} low-confidence predictions...", flush=True)
                low_conf_df = df.loc[low_conf_mask].copy()
                updated = low_conf_df.apply(self._keyword_fallback, axis=1)
                cols_to_update = ['predicted_label_id', 'predicted_label_name', 'prediction_confidence', 'prediction_source']
                df.loc[updated.index, cols_to_update] = updated[cols_to_update].values
        
        print(f"\n✅ Prediction complete!")
        print(f"   Average confidence: {df['prediction_confidence'].mean():.3f}")
        print(f"   Min confidence: {df['prediction_confidence'].min():.3f}")
        
        print(f"\n📊 PREDICTION SOURCE BREAKDOWN:")
        print(df['prediction_source'].value_counts())
        
        print(f"\n📊 TOPIC DISTRIBUTION:")
        print(df['predicted_label_name'].value_counts())
        
        return df
    
    def _keyword_fallback(self, row):
        """Apply keyword matching for a single row and return an updated row"""
        text = row.get('combined_text', '')
        if isinstance(text, str):
            text = text.lower()
        else:
            text = ''
        lang = row.get('language_code', 'en')
        
        topic_scores = {}
        for topic_name, lang_keywords in self.multilingual_keywords.items():
            keywords = lang_keywords.get(lang, lang_keywords.get('en', []))
            score = sum(1 for kw in keywords if kw in text)
            topic_scores[topic_name] = score
        
        if max(topic_scores.values()) > 0:
            best_topic = max(topic_scores, key=topic_scores.get)
            total = sum(topic_scores.values())
            confidence = topic_scores[best_topic] / (total + 1e-6)
            
            label_to_id = {v: k for k, v in self.topic_labels.items()}
            row['predicted_label_name'] = best_topic
            row['predicted_label_id'] = label_to_id[best_topic]
            row['prediction_confidence'] = confidence
            row['prediction_source'] = 'keyword_fallback'
        return row
    
    def save_model(self, path='topic_labeler_model.pkl'):
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("No trained model to save")
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='topic_labeler_model.pkl'):
        """Load trained model from disk"""
        self.model = joblib.load(path)
        self.trained = True
        print(f"✅ Model loaded from {path}")
    
    def export_for_review(self, df, output_path, n_samples=100, 
                         strategy='lowest_confidence', include_disagreements=True):
        """
        Export articles for manual review to improve the model
        
        Parameters:
        -----------
        df : DataFrame
            Predictions from predict() method
        output_path : str/Path
            Where to save the CSV for review
        n_samples : int
            Number of samples to export
        strategy : str
            'lowest_confidence' - pick articles with lowest prediction confidence
            'random_stratified' - random sample from each topic
            'uncertainty_sampling' - articles where top 2 predictions are close
        include_disagreements : bool
            If df has 'old_label' column, prioritize disagreements
        """
        print("\n" + "="*80)
        print("EXPORTING ARTICLES FOR MANUAL REVIEW")
        print("="*80)
        
        df = df.copy()
        
        # Strategy 1: Prioritize disagreements with old labels
        if include_disagreements and 'old_label' in df.columns:
            disagreements = df[df['predicted_label_name'] != df['old_label']]
            if len(disagreements) > 0:
                n_disagree = min(n_samples // 3, len(disagreements))
                print(f"📌 Including {n_disagree} disagreements with old labels")
                review_set = disagreements.nlargest(n_disagree, 'prediction_confidence')
                remaining = n_samples - n_disagree
            else:
                review_set = pd.DataFrame()
                remaining = n_samples
        else:
            review_set = pd.DataFrame()
            remaining = n_samples
        
        # Strategy 2: Select based on chosen strategy
        available_df = df[~df.index.isin(review_set.index)]
        
        if strategy == 'lowest_confidence':
            print(f"📌 Selecting {remaining} lowest-confidence predictions")
            selected = available_df.nsmallest(remaining, 'prediction_confidence')
        
        elif strategy == 'uncertainty_sampling':
            print(f"📌 Selecting {remaining} most uncertain predictions (top 2 classes close)")
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                X = available_df['processed_text']
                probs = self.model.predict_proba(X)
                # Calculate margin between top 2 predictions
                sorted_probs = np.sort(probs, axis=1)
                margins = sorted_probs[:, -1] - sorted_probs[:, -2]
                available_df['uncertainty_margin'] = margins
                selected = available_df.nsmallest(remaining, 'uncertainty_margin')
            else:
                # Fallback to lowest confidence
                selected = available_df.nsmallest(remaining, 'prediction_confidence')
        
        elif strategy == 'random_stratified':
            print(f"📌 Selecting {remaining} random samples (stratified by topic)")
            # Sample proportionally from each topic
            selected = available_df.groupby('predicted_label_name', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, remaining * len(x) // len(available_df))))
            ).head(remaining)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Combine
        review_set = pd.concat([review_set, selected])
        
        # Prepare output columns
        output_cols = ['title', 'description', 'predicted_label_name', 
                      'prediction_confidence', 'prediction_source']
        
        # Add language if available
        if 'language_code' in review_set.columns:
            output_cols.insert(0, 'language_code')
        
        # Add old label for comparison if available
        if 'old_label' in review_set.columns:
            output_cols.append('old_label')
        
        # Add URL if available
        if 'url' in review_set.columns:
            output_cols.append('url')
        
        # Add empty column for manual correction
        review_set['manual_label'] = ''
        review_set['review_notes'] = ''
        output_cols.extend(['manual_label', 'review_notes'])
        
        # Filter to available columns
        output_cols = [col for col in output_cols if col in review_set.columns]
        
        # Save
        review_set[output_cols].to_csv(output_path, index=False)
        
        print(f"\n✅ Exported {len(review_set)} articles to {output_path}")
        print(f"\n📊 REVIEW SET BREAKDOWN:")
        print(review_set['predicted_label_name'].value_counts())
        print(f"\n💡 Instructions:")
        print(f"   1. Open {output_path}")
        print(f"   2. Fill in 'manual_label' column with correct topic (0-5)")
        print(f"   3. Optionally add notes in 'review_notes' column")
        print(f"   4. Save and use in next training iteration")
        
        return review_set
        ax1.axvline(df['prediction_confidence'].median(), color='orange', linestyle='--',
                   label=f'Median: {df["prediction_confidence"].median():.3f}')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Number of Articles')
        ax1.set_title('Overall Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence by Topic (Boxplot)
        ax2 = axes[0, 1]
        topic_order = df.groupby('predicted_label_name')['prediction_confidence'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, y='predicted_label_name', x='prediction_confidence', 
                   order=topic_order, ax=ax2, palette='Set2')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Topic')
        ax2.set_title('Confidence Distribution by Topic')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Topic Distribution (Bar Chart)
        ax3 = axes[0, 2]
        topic_counts = df['predicted_label_name'].value_counts()
        colors = sns.color_palette('Set3', len(topic_counts))
        topic_counts.plot(kind='bar', ax=ax3, color=colors, edgecolor='black')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Number of Articles')
        ax3.set_title('Article Distribution by Topic')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, v in enumerate(topic_counts):
            ax3.text(i, v + max(topic_counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence Thresholds Analysis
        ax4 = axes[1, 0]
        thresholds = np.arange(0, 1.01, 0.05)
        articles_above = [((df['prediction_confidence'] >= t).sum()) for t in thresholds]
        ax4.plot(thresholds, articles_above, marker='o', linewidth=2, markersize=4, color='darkgreen')
        ax4.fill_between(thresholds, articles_above, alpha=0.3, color='lightgreen')
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Number of Articles')
        ax4.set_title('Articles Above Confidence Threshold')
        ax4.grid(True, alpha=0.3)
        
        # Add reference lines
        for threshold, label in [(0.4, '0.4'), (0.5, '0.5'), (0.7, '0.7')]:
            count = (df['prediction_confidence'] >= threshold).sum()
            ax4.axvline(threshold, color='red', linestyle='--', alpha=0.5)
            ax4.text(threshold, max(articles_above)*0.95, 
                    f'{threshold}\n({count})', ha='center', fontsize=8)
        
        # 5. Prediction Source Breakdown (if available)
        ax5 = axes[1, 1]
        if 'prediction_source' in df.columns:
            source_counts = df['prediction_source'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(source_counts)]
            wedges, texts, autotexts = ax5.pie(source_counts, labels=source_counts.index, 
                                               autopct='%1.1f%%', colors=colors,
                                               startangle=90, textprops={'fontsize': 10})
            ax5.set_title('Prediction Source Distribution')
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax5.text(0.5, 0.5, 'No prediction source data', ha='center', va='center', fontsize=12)
            ax5.axis('off')
        
        # 6. Confidence vs Topic Heatmap
        ax6 = axes[1, 2]
        confidence_bins = pd.cut(df['prediction_confidence'], bins=[0, 0.4, 0.6, 0.8, 1.0],
                                labels=['Low\n(0-0.4)', 'Medium\n(0.4-0.6)', 
                                       'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)'])
        heatmap_data = pd.crosstab(df['predicted_label_name'], confidence_bins)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax6, 
                   cbar_kws={'label': 'Article Count'})
        ax6.set_xlabel('Confidence Level')
        ax6.set_ylabel('Topic')
        ax6.set_title('Topic vs Confidence Level Heatmap')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plot_file = Path(save_path) / 'prediction_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to {plot_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("CONFIDENCE STATISTICS")
        print("="*80)
        print(f"Mean confidence: {df['prediction_confidence'].mean():.3f}")
        print(f"Median confidence: {df['prediction_confidence'].median():.3f}")
        print(f"Std deviation: {df['prediction_confidence'].std():.3f}")
        print(f"\nArticles by confidence level:")
        print(f"  Very High (>0.8): {(df['prediction_confidence'] > 0.8).sum()} ({(df['prediction_confidence'] > 0.8).sum()/len(df)*100:.1f}%)")
        print(f"  High (0.6-0.8): {((df['prediction_confidence'] >= 0.6) & (df['prediction_confidence'] <= 0.8)).sum()} ({((df['prediction_confidence'] >= 0.6) & (df['prediction_confidence'] <= 0.8)).sum()/len(df)*100:.1f}%)")
        print(f"  Medium (0.4-0.6): {((df['prediction_confidence'] >= 0.4) & (df['prediction_confidence'] < 0.6)).sum()} ({((df['prediction_confidence'] >= 0.4) & (df['prediction_confidence'] < 0.6)).sum()/len(df)*100:.1f}%)")
        print(f"  Low (<0.4): {(df['prediction_confidence'] < 0.4).sum()} ({(df['prediction_confidence'] < 0.4).sum()/len(df)*100:.1f}%)")

    def visualize_predictions(self, df, save_path=None, show=False):
        """
        Create visualizations of prediction quality
        
        Parameters:
        -----------
        df : DataFrame
            Results from predict() method
        save_path : str/Path (optional)
            If provided, save plots to this directory
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("⚠️  matplotlib/seaborn not installed. Run: pip install matplotlib seaborn")
            return
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Topic Classification Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confidence Distribution (Overall)
        ax1 = axes[0, 0]
        ax1.hist(df['prediction_confidence'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(df['prediction_confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["prediction_confidence"].mean():.3f}')
        ax1.axvline(df['prediction_confidence'].median(), color='orange', linestyle='--',
                   label=f'Median: {df["prediction_confidence"].median():.3f}')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Number of Articles')
        ax1.set_title('Overall Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence by Topic (Boxplot)
        ax2 = axes[0, 1]
        topic_order = df.groupby('predicted_label_name')['prediction_confidence'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, y='predicted_label_name', x='prediction_confidence', 
                   order=topic_order, ax=ax2, palette='Set2')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Topic')
        ax2.set_title('Confidence Distribution by Topic')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Topic Distribution (Bar Chart)
        ax3 = axes[0, 2]
        topic_counts = df['predicted_label_name'].value_counts()
        colors = sns.color_palette('Set3', len(topic_counts))
        topic_counts.plot(kind='bar', ax=ax3, color=colors, edgecolor='black')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Number of Articles')
        ax3.set_title('Article Distribution by Topic')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, v in enumerate(topic_counts):
            ax3.text(i, v + max(topic_counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 4. Confidence Thresholds Analysis
        ax4 = axes[1, 0]
        thresholds = np.arange(0, 1.01, 0.05)
        articles_above = [((df['prediction_confidence'] >= t).sum()) for t in thresholds]
        ax4.plot(thresholds, articles_above, marker='o', linewidth=2, markersize=4, color='darkgreen')
        ax4.fill_between(thresholds, articles_above, alpha=0.3, color='lightgreen')
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Number of Articles')
        ax4.set_title('Articles Above Confidence Threshold')
        ax4.grid(True, alpha=0.3)
        
        # Add reference lines
        for threshold, label in [(0.4, '0.4'), (0.5, '0.5'), (0.7, '0.7')]:
            count = (df['prediction_confidence'] >= threshold).sum()
            ax4.axvline(threshold, color='red', linestyle='--', alpha=0.5)
            ax4.text(threshold, max(articles_above)*0.95, 
                    f'{threshold}\n({count})', ha='center', fontsize=8)
        
        # 5. Prediction Source Breakdown (if available)
        ax5 = axes[1, 1]
        if 'prediction_source' in df.columns:
            source_counts = df['prediction_source'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(source_counts)]
            wedges, texts, autotexts = ax5.pie(source_counts, labels=source_counts.index, 
                                               autopct='%1.1f%%', colors=colors,
                                               startangle=90, textprops={'fontsize': 10})
            ax5.set_title('Prediction Source Distribution')
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax5.text(0.5, 0.5, 'No prediction source data', ha='center', va='center', fontsize=12)
            ax5.axis('off')
        
        # 6. Confidence vs Topic Heatmap
        ax6 = axes[1, 2]
        confidence_bins = pd.cut(df['prediction_confidence'], bins=[0, 0.4, 0.6, 0.8, 1.0],
                                labels=['Low\n(0-0.4)', 'Medium\n(0.4-0.6)', 
                                       'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)'])
        heatmap_data = pd.crosstab(df['predicted_label_name'], confidence_bins)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax6, 
                   cbar_kws={'label': 'Article Count'})
        ax6.set_xlabel('Confidence Level')
        ax6.set_ylabel('Topic')
        ax6.set_title('Topic vs Confidence Level Heatmap')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plot_file = Path(save_path) / 'prediction_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to {plot_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("CONFIDENCE STATISTICS")
        print("="*80)
        print(f"Mean confidence: {df['prediction_confidence'].mean():.3f}")
        print(f"Median confidence: {df['prediction_confidence'].median():.3f}")
        print(f"Std deviation: {df['prediction_confidence'].std():.3f}")
        print(f"\nArticles by confidence level:")
        print(f"  Very High (>0.8): {(df['prediction_confidence'] > 0.8).sum()} ({(df['prediction_confidence'] > 0.8).sum()/len(df)*100:.1f}%)")
        print(f"  High (0.6-0.8): {((df['prediction_confidence'] >= 0.6) & (df['prediction_confidence'] <= 0.8)).sum()} ({((df['prediction_confidence'] >= 0.6) & (df['prediction_confidence'] <= 0.8)).sum()/len(df)*100:.1f}%)")
        print(f"  Medium (0.4-0.6): {((df['prediction_confidence'] >= 0.4) & (df['prediction_confidence'] < 0.6)).sum()} ({((df['prediction_confidence'] >= 0.4) & (df['prediction_confidence'] < 0.6)).sum()/len(df)*100:.1f}%)")
        print(f"  Low (<0.4): {(df['prediction_confidence'] < 0.4).sum()} ({(df['prediction_confidence'] < 0.4).sum()/len(df)*100:.1f}%)")


def save_metrics(model_name, metrics, metrics_path):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'model': model_name,
        'cv_accuracy_mean': metrics.get('cv_accuracy_mean'),
        'cv_accuracy_std': metrics.get('cv_accuracy_std'),
        'cv_precision_mean': metrics.get('cv_precision_mean'),
        'cv_precision_std': metrics.get('cv_precision_std'),
        'train_accuracy': metrics.get('train_accuracy'),
        'train_precision_weighted': metrics.get('train_precision_weighted')
    }
    try:
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(metrics_path, index=False)
        print(f"\n✅ Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"\n❌ Error saving metrics: {e}")


# ============================================================================
# USAGE EXAMPLE - COMBINING MANUAL + KEYWORD LABELS
# ============================================================================

if __name__ == "__main__":
    # Paths
    DATA_DIR = Path(__file__).parent.parent / "datascience" / "data_africa"
    
    MANUAL_LABELED_FILE = DATA_DIR / "manual_training_set.csv"  # Your 200 manual labels
    KEYWORD_LABELED_FILE = DATA_DIR / "all_languages_labeled.csv"  # Output from old code
    UNLABELED_FILE = DATA_DIR / "all_languages_links_ok.csv"
    OUTPUT_FILE = DATA_DIR / "ldamulti.csv"
    REVIEW_FILE = DATA_DIR / "for_manual_review.csv"
    VIZ_DIR = DATA_DIR / "visualizations"
    
    # ========================================================================
    # STEP 1: Load manually labeled data (200 examples)
    # ========================================================================
    print("📂 Loading manually labeled data...")
    manual_df = pd.read_csv(MANUAL_LABELED_FILE)
    # Expected columns: text, language_code, label (0-5)
    # Normalize manual text column name for downstream processing
    if 'text' not in manual_df.columns:
        raise ValueError("manual_training_set.csv must contain a 'text' column")
    manual_df['text'] = manual_df['text'].fillna('')
    print(f"   Loaded {len(manual_df)} manual labels\n")
    
    # ========================================================================
    # STEP 2: Load keyword-labeled data from your previous code
    # ========================================================================
    print("📂 Loading keyword-labeled data...")
    keyword_df = pd.read_csv(KEYWORD_LABELED_FILE)
    # Expected columns: title, description, language_code, 
    #                   predicted_label_id, predicted_label_name, prediction_confidence
    keyword_df['text'] = keyword_df[['title', 'description']].fillna('').agg(' '.join, axis=1)
    print(f"   Loaded {len(keyword_df)} keyword labels")
    print(f"   Confidence stats: mean={keyword_df['prediction_confidence'].mean():.3f}, "
          f"median={keyword_df['prediction_confidence'].median():.3f}\n")
    
    # ========================================================================
    # STEP 3: Train model on BOTH manual + high-confidence keyword labels
    # ========================================================================
    labeler = SupervisedTopicLabeler()
    
    # Try different thresholds to see what works best
    # Higher threshold = fewer but higher quality keyword labels
    # Lower threshold = more training data but potentially noisier
    KEYWORD_CONFIDENCE_THRESHOLD = 0.5  # Adjust this (try 0.4, 0.5, 0.6)
    
    labeler.train(
        manual_df=manual_df,
        keyword_df=keyword_df,
        keyword_confidence_threshold=KEYWORD_CONFIDENCE_THRESHOLD,
        text_cols=['text'],
        manual_label_col='label',  # Manual ground truth column
        keyword_label_col='predicted_label_id',  # From old code
        keyword_conf_col='prediction_confidence'  # From old code
    )
    
    # ========================================================================
    # STEP 4: Save model for reuse
    # ========================================================================
    labeler.save_model(DATA_DIR / 'topic_model_semisupervised.pkl')
    
    # ========================================================================
    # STEP 5: Predict on all data (or new unlabeled data)
    # ========================================================================
    print(f"\n📂 Loading data to label from {UNLABELED_FILE}...")
    unlabeled_df = pd.read_csv(UNLABELED_FILE)
    
    results_df = labeler.predict(
        unlabeled_df,
        text_cols=['title', 'description'],
        confidence_threshold=0.5,      # Requested threshold
        use_keyword_fallback=True
    )
    
    # ========================================================================
    # STEP 6: Save results
    # ========================================================================
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Labeled data saved to {OUTPUT_FILE}")
    
    # ========================================================================
    # STEP 7: Compare with original keyword labels
    # ========================================================================
    if 'predicted_label_id' in keyword_df.columns:
        print("\n" + "="*80)
        print("COMPARISON: NEW MODEL vs ORIGINAL KEYWORD LABELS")
        print("="*80)
        
        # Merge on a unique identifier (adjust if you have a different ID column)
        # Assuming row order is preserved
        comparison = results_df[['predicted_label_name', 'prediction_confidence']].copy()
        comparison.columns = ['new_label', 'new_confidence']
        comparison['old_label'] = keyword_df['predicted_label_name'].values[:len(comparison)]
        comparison['old_confidence'] = keyword_df['prediction_confidence'].values[:len(comparison)]
        
        # Add to results for export
        results_df['old_label'] = comparison['old_label']
        
        # Check agreement
        comparison['labels_match'] = comparison['new_label'] == comparison['old_label']
        agreement_rate = comparison['labels_match'].mean()
        
        print(f"\n📊 Agreement with original keyword labels: {agreement_rate:.1%}")
        print(f"   ({comparison['labels_match'].sum()} / {len(comparison)} articles)")
        
        # Show disagreements
        disagreements = comparison[~comparison['labels_match']]
        if len(disagreements) > 0:
            print(f"\n⚠️  {len(disagreements)} disagreements found")
            print("\nTop 10 disagreements:")
            print(disagreements[['old_label', 'new_label', 'old_confidence', 'new_confidence']].head(10))
    
    # ========================================================================
    # STEP 8: Generate visualizations
    # ========================================================================
    labeler.visualize_predictions(results_df, save_path=VIZ_DIR, show=False)
    
    # ========================================================================
    # STEP 9: Export articles for manual review (iterative improvement)
    # ========================================================================
    review_df = labeler.export_for_review(
        results_df,
        output_path=REVIEW_FILE,
        n_samples=100,  # Number of articles to review
        strategy='lowest_confidence',  # Options: 'lowest_confidence', 'uncertainty_sampling', 'random_stratified'
        include_disagreements=True  # Prioritize disagreements with old labels
    )

    save_metrics(
        model_name="ldamulti_it-idf",
        metrics=labeler.metrics,
        metrics_path=DATA_DIR / "model_metrics.csv"
    )
    
    # ========================================================================
    # STEP 10: Instructions for iterative improvement
    # ========================================================================
    print("\n" + "="*80)
    print("🔄 ITERATIVE IMPROVEMENT WORKFLOW")
    print("="*80)
    print(f"""
1. Review the exported file: {REVIEW_FILE}
2. Fill in the 'manual_label' column with correct labels (0-5)
3. Save the reviewed file
4. Load it and combine with your existing manual labels:
   
   ```python
   # Load your new reviews
   new_reviews = pd.read_csv('{REVIEW_FILE}')
   new_reviews = new_reviews[new_reviews['manual_label'].notna()]
   
   # Combine with existing manual labels
   existing_manual = pd.read_csv('{MANUAL_LABELED_FILE}')
   combined_manual = pd.concat([existing_manual, new_reviews])
   combined_manual.to_csv('{MANUAL_LABELED_FILE}', index=False)
   
   # Retrain the model
   labeler = SupervisedTopicLabeler()
   labeler.train(manual_df=combined_manual, keyword_df=keyword_df)
   ```

5. Repeat this process until prediction quality is satisfactory

📊 CURRENT STATUS:
   - Training data: {len(manual_df)} manual + high-confidence keyword labels
   - Unlabeled articles: {len(results_df)}
   - Ready for review: {len(review_df)}
   - Visualizations saved to: {VIZ_DIR}
    """)