"""RESULTS 58 step 2a: the training QUESTIONS for the phone judge, in the forms the product asks.

Families (fam):
  object   "Is there a real X anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
  subject  "Is this a photo of X?"                                  (planner subject form; product cut 0.999)
  beach    "Is a beach clearly visible in this photo?"              (RESULTS 47 grounding)
  person   "Is the person in the red box X?" / "Does the person in the red box look X?"   (person crop + red box)
  photo    photo-level looks ("Is this photo blurry?", "Is this a black and white photo?", ...)
  free     planner-written questions: the 10-05 distillation plans (planner27ball_clean.jsonl, synthetic fuzz requests),
           the complex_bench TRAIN SFT plans (cb_train_v3, train libraries' requests only) and the TRAIN oracle questions
           (complex_bench/oracle/<train user>.parquet). Nothing from dev/test requests.
Each row also carries 'look' = the text used to retrieve likely-yes photos (PE-Core text vector).
Usage (CPU): python eval/judge_teach/questions.py -> data/public/judge_teach/questions.jsonl
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/public/judge_teach"
REAL = "Is there a real {x} anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"

OBJECTS = """dog,puppy,cat,kitten,horse,cow,sheep,goat,pig,chicken,duck,goose,swan,pigeon,seagull,parrot,owl,eagle,bird,
fish,turtle,frog,snake,lizard,butterfly,bee,spider,squirrel,rabbit,deer,fox,bear,elephant,giraffe,zebra,lion,tiger,monkey,
camel,dolphin,whale,seal,penguin,kangaroo,car,taxi,bus,truck,van,motorcycle,scooter,bicycle,tricycle,skateboard,
train,tram,airplane,helicopter,hot air balloon,boat,sailboat,canoe,kayak,ship,ferry,yacht,jet ski,tractor,fire truck,
police car,ambulance,limousine,camper van,wheelchair,stroller,shopping cart,traffic light,stop sign,street sign,bench,
fountain,statue,umbrella,parasol,tent,flag,kite,balloon,ball,football,basketball,tennis racket,golf club,surfboard,
snowboard,skis,sled,swing,slide,trampoline,guitar,piano,violin,drum,microphone,trumpet,camera,laptop,mobile phone,
television,clock,lamp,candle,book,newspaper,map,backpack,suitcase,handbag,wallet,hat,cap,helmet,sunglasses,glasses,
watch,necklace,ring,shoe,boot,sandal,scarf,glove,tie,dress,suit,jacket,coat,umbrella stroller,chair,sofa,table,bed,
desk,shelf,bookcase,window,door,staircase,fireplace,mirror,painting,poster,bottle,wine glass,beer glass,mug,coffee cup,
teapot,plate,bowl,fork,knife,spoon,chopsticks,pizza,burger,sandwich,hot dog,sushi,cake,birthday cake,cupcake,cookie,
donut,bread,croissant,pastry,pie,ice cream,chocolate,candy,salad,soup,pasta,noodles,rice,steak,sausage,egg,cheese,
fruit,apple,banana,orange,lemon,strawberry,grapes,watermelon,pineapple,tomato,carrot,broccoli,corn,potato,mushroom,
french fries,cocktail,beer,wine,coffee,tea,juice,flower,rose,sunflower,tulip,tree,palm tree,christmas tree,cactus,
houseplant,flower pot,pumpkin,snowman,fireworks,campfire,barbecue grill,swimming pool,hot tub,lighthouse,windmill,bridge,
tower,castle,church,temple,mosque,skyscraper,house,barn,cabin,pier,dock,fence,road sign,mailbox,fire hydrant,
bus stop,phone booth,graffiti,mural,sculpture,trophy,medal,gift,wrapped present,toy,teddy bear,doll,lego,robot,drone,
baby,stuffed animal,crown,mask,wig,costume,uniform,tattoo,wedding dress,veil,bouquet,cake stand,menu,receipt,ticket,
passport,whiteboard,computer,keyboard,headphones,speaker,guitar case,easel,tripod,binoculars,fishing rod,anchor,rope,
ladder,wheelbarrow,shovel,axe,hammer,saw,lawn mower,vacuum cleaner,washing machine,refrigerator,oven,microwave,sink,
toilet,bathtub,shower,towel,pillow,blanket,carpet,curtain,vase,basket,box,bag,trash can,recycling bin""".replace("\n", "").split(",")

SUBJECTS = """food,a beach,a sunset,a sunrise,flowers,a cat,a dog,a church,a mountain,mountains,a lake,a river,a waterfall,the sea,
the ocean,a forest,a desert,a canyon,a glacier,a volcano,a cave,a field,a meadow,a garden,a park,a city skyline,a city at
night,a street,a market,a harbour,a marina,an island,a coastline,a cliff,a rocky shore,a lake shore,snow,a snowy landscape,
fog,mist,rain,a storm,lightning,a rainbow,clouds,the moon,stars,the night sky,the northern lights,fireworks,a bonfire,
a castle,a palace,a temple,a mosque,a cathedral,a church interior,a museum,an art gallery,a stadium,a concert,a stage,
a festival,a parade,a wedding,a birthday party,a party,a dinner,a picnic,a barbecue,a restaurant,a cafe,a bar,a kitchen,
a living room,a bedroom,a hotel room,an office,a classroom,a library,a shop,a supermarket,a playground,a zoo,an aquarium,
a farm,a vineyard,a bridge,a lighthouse,a train station,an airport,an airplane window view,a road trip,a hike,a camping
trip,a ski trip,skiing,surfing,swimming,cycling,running,a football game,a basketball game,a baseball game,a tennis match,
a golf course,a swimming pool,a boat trip,a sailing boat,a cruise ship,a car,a motorcycle,a train,a bicycle,a horse,
a bird,birds,wildlife,an animal,a pet,a baby,children playing,a family,a group of people,a crowd,a selfie,a portrait,
a landscape,architecture,a building,ruins,a monument,a statue,street art,graffiti,a sign,a document,a screenshot,a receipt,
a menu,a map,a whiteboard,text,a painting,a drawing,a poster,a dish,a meal,a cake,dessert,breakfast,coffee,drinks,
cocktails,wine,beer,fruit,vegetables,a cityscape,a village,a countryside,autumn leaves,spring blossoms,cherry blossoms,
a sunflower field,a tulip field,a christmas tree,christmas decorations,halloween,a pumpkin patch,a snowman,the beach at
sunset,golden hour light,a reflection,a silhouette,a close-up,a macro shot,an aerial view,a panorama,underwater""".replace("\n", "").split(",")

PERSON = """smiling,laughing,wearing sunglasses,wearing glasses,wearing a hat,wearing a cap,wearing a baseball cap,
wearing a beanie,wearing a sun hat,wearing a cowboy hat,wearing a helmet,wearing a bike helmet,wearing a conical hat,
wearing a headscarf,wearing a turban,wearing a crown,wearing a wig,wearing a costume,wearing a uniform,wearing a suit,
wearing a tie,wearing a bow tie,wearing a dress,wearing a wedding dress,wearing a swimsuit,wearing shorts,wearing jeans,
wearing a jacket,wearing a coat,wearing a scarf,wearing gloves,wearing a backpack,wearing a necklace,wearing earrings,
wearing a watch,wearing a mask,wearing a lanyard,wearing a life jacket,wearing a wetsuit,wearing ski goggles,
wearing a red jacket,wearing a blue jacket,wearing a black jacket,wearing a yellow jacket,wearing a green jacket,
wearing a red shirt,wearing a blue shirt,wearing a white shirt,wearing a black shirt,wearing a green shirt,wearing a pink
shirt,wearing a striped shirt,wearing a red dress,wearing a white dress,wearing a black dress,wearing a blue dress,
wearing a pink dress,wearing a red hat,wearing a blue cap,wearing a white cap,wearing a black cap,wearing a red cap,
wearing a blue beanie,wearing a pink helmet,wearing a white helmet,wearing a yellow tie,wearing a red scarf,
wearing an orange shirt,wearing a purple shirt,wearing a grey sweater,wearing a hoodie,wearing a t-shirt,wearing a vest,
wearing a sari,wearing traditional clothing,wearing a kimono,wearing a red traditional wrap,wearing a football jersey,
wearing a graduation gown,wearing a chef's hat,wearing a santa hat,wearing face paint,holding a microphone,singing,
giving a speech,holding a drink,holding a beer,holding a glass of wine,holding a cup of coffee,holding a phone,
taking a photo,holding a camera,holding an umbrella,holding a baby,holding a child,holding a dog,holding a cat,
holding flowers,holding a bouquet,holding a book,holding a map,holding a sign,holding a flag,holding a trophy,
holding a medal,holding a gift,holding a cake,holding a knife,holding a guitar,playing the guitar,playing the piano,
playing drums,playing a violin,holding a tennis racket,playing tennis,holding a ball,playing football,playing basketball,
holding a fish,fishing,holding a surfboard,surfing,swimming,in the water,skiing,snowboarding,ice skating,cycling,
riding a bike,riding a horse,riding a motorcycle,driving a car,on a boat,in a kayak,rowing,hiking,climbing,running,
jumping,dancing,eating,drinking,cooking,reading,writing,sleeping,sitting,lying down,standing,walking,waving,hugging someone,
kissing someone,pointing,clapping,blowing out candles,cutting a cake,opening a present,on stage,at a concert,at a wedding,
at a party,at a restaurant,at a table,at the beach,in the snow,in the rain,outdoors,indoors,in a car,in a city,in a forest,
on a mountain,at a waterfall,by a lake,by the sea,in front of a monument,in a museum,in a kitchen,in an office,in a pool,
on a beach chair,at sunset,at night,with a beard,with a mustache,clean-shaven,bald,with long hair,with short hair,
with curly hair,with blond hair,with red hair,with grey hair,with a ponytail,with a tattoo,with a dog,with a cat,
with a child,with a group of friends,alone in the photo,in a wheelchair,on crutches,in a costume,in a team photo,
shirtless,in underwear,carrying a backpack,carrying a bag,carrying something on the head,carrying a child on the shoulders,
pushing a stroller,smoking,wearing makeup,with the eyes closed,looking at the camera,making a funny face,
giving a thumbs up,wearing a party hat,wearing a lei,wearing a medal,wearing a race number,wearing a name tag""".replace("\n", "").split(",")

PERSON_LOOKS = ["tired", "happy", "sad", "angry", "surprised", "excited", "relaxed", "serious", "sunburnt", "cold",
                "heavier", "fit", "older", "young", "sick", "bored", "nervous", "proud", "sweaty", "dressed up"]

PHOTO = ["Is this photo blurry?", "Is this photo out of focus?", "Is there motion blur in this photo?",
         "Is this photo overexposed?", "Is this photo underexposed or very dark?", "Is this a black and white photo?",
         "Is this photo in sepia tones?", "Is this photo taken at night?", "Is this photo taken indoors?",
         "Is this photo taken outdoors?", "Is this a selfie?", "Is this a screenshot?", "Is this a photo of a screen?",
         "Is this a photo of a document?", "Is this a photo of a photo?", "Is this a close-up photo?",
         "Is this a wide landscape shot?", "Is this an aerial photo?", "Is this a panoramic photo?",
         "Is this photo taken underwater?", "Is this a group photo?", "Is this a portrait photo of one person?",
         "Is this photo taken during golden hour?", "Is this photo taken in the snow?", "Is the sky blue in this photo?",
         "Is it raining in this photo?", "Is it foggy in this photo?", "Is this a photo with a shallow depth of field?",
         "Is this photo taken from a moving vehicle?", "Is this a long-exposure photo?", "Is this photo grainy or noisy?",
         "Is there lens flare in this photo?", "Is this a vintage-looking photo?", "Is this a photo with a filter applied?",
         "Is this a reflection photo?", "Is this a silhouette photo?", "Is this photo taken at dusk?",
         "Is this photo taken at sunrise?", "Is there purple lighting in this photo?", "Is there blue lighting in this photo?",
         "Is there red lighting in this photo?", "Is there green lighting in this photo?", "Is there neon lighting in this photo?",
         "Is the water turquoise in this photo?", "Is there golden sunlight in this photo?", "Is there smog or haze in this photo?",
         "Is there a lot of red in this photo?", "Is there a lot of yellow in this photo?", "Is the photo mostly blue?",
         "Are there bright colours in this photo?", "Is this photo very colourful?", "Is the main subject pink?",
         "Is the main subject purple?", "Is the main subject orange?", "Is there something magenta in this photo?"]

PREFIX = re.compile(r"(?i)^(is|are|does|do|was|were|can you see|could)\b\s*(there|this|the|it|a|an|any)?\b\s*"
                    r"(a\s+)?(real\s+)?(photo|picture|video|image)?\b\s*(of|with|showing|where|that)?\b\s*")


def look_of(q):
    t = PREFIX.sub("", q.strip()).rstrip("?")
    t = re.sub(r"(?i)\s+(anywhere\s+)?in this (photo|picture|image|video).*$", "", t)
    t = re.sub(r"(?i)\(not a drawing.*$", "", t).strip()
    return t or q


def planner_questions():
    out = set()
    for f in [ROOT / "data/public/distill/planner27ball_clean.jsonl", ROOT / "data/public/distill/cb_train_v3/rows_fixed.jsonl"]:
        for line in open(f):
            r = json.loads(line)
            if r.get("split", "train") != "train":
                continue
            m = re.search(r"\{.*\}", r["output"], re.S)
            try:
                P = json.loads(m.group(0))
            except Exception:
                continue
            for a in P.get("albums", []):
                qs = [a.get(k) for k in ("judge_question", "filter_question", "exclude_question")]
                qs += [(a.get(k) or {}).get("judge_question") for k in ("anchor", "until")]
                for q in qs:
                    q = (q or "").strip()
                    if q and 8 < len(q) < 200 and q.endswith("?") and " AND " not in q:
                        out.add((q, a.get("person") is not None and "red box" in q.lower()))
    return out


def oracle_questions():
    sp = json.load(open(ROOT / "eval/complex_bench/splits.json"))
    import pandas as pd
    out = set()
    for u in sp["train"]:
        f = ROOT / f"eval/complex_bench/oracle/{u}.parquet"
        if f.exists():
            for q in pd.read_parquet(f, columns=["q"]).q.unique():
                if q.startswith("PERSON["):
                    q = q.split("::", 1)[1]
                    q = re.sub(r"(?i)^(is|does|are) the person\b", lambda m: m.group(1) + " the person in the red box", q)
                    out.add((q, True))
                else:
                    out.add((q, False))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for x in OBJECTS:
        x = x.strip()
        if x:
            rows.append(dict(q=REAL.format(x=x), fam="object", look=f"a photo with a {x}"))
    for x in SUBJECTS:
        x = x.strip()
        if x:
            rows.append(dict(q=f"Is this a photo of {x}?", fam="subject", look=f"a photo of {x}"))
    rows.append(dict(q="Is a beach clearly visible in this photo?", fam="beach", look="a photo of a beach"))
    for x in PERSON:
        x = x.strip()
        if x:
            rows.append(dict(q=f"Is the person in the red box {x}?", fam="person", look=f"a photo of a person {x}"))
    for x in PERSON_LOOKS:
        rows.append(dict(q=f"Does the person in the red box look {x}?", fam="person", look=f"a person looking {x}"))
    for q in PHOTO:
        rows.append(dict(q=q, fam="photo", look=look_of(q)))
    have = {r["q"] for r in rows}
    for q, person in sorted(planner_questions() | oracle_questions()):
        if q in have:
            continue
        have.add(q)
        if "red box" in q.lower():
            rows.append(dict(q=q, fam="person", look="a person " + look_of(q.lower().replace("the person in the red box", "")).lstrip()))
        else:
            rows.append(dict(q=q, fam="free", look=look_of(q)))
    with open(OUT / "questions.jsonl", "w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps(dict(qid=i, **r)) + "\n")
    from collections import Counter
    print(len(rows), Counter(r["fam"] for r in rows))


if __name__ == "__main__":
    main()
