"""Vector Database Visualized.

A tiny, fully-local Streamlit demo that turns short text chunks into TF-IDF
vectors, projects them into an interactive 3D space, and lets you run a semantic
search to see how a vector database finds "nearest neighbors". No API keys, no
internet, no server.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Sample data: a tiny "database" of text chunks across a few themes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    title: str
    category: str
    text: str


CHUNKS: list[Chunk] = [
    # Animals & Wildlife
    Chunk("Honeybees", "Animals & Wildlife",
          "Honeybees collect nectar from flowers and communicate the way to food with a waggle dance."),
    Chunk("Migrating birds", "Animals & Wildlife",
          "Many birds fly thousands of miles each season, navigating by the sun, stars, and landmarks."),
    Chunk("Elephant herds", "Animals & Wildlife",
          "Elephants live in close family herds led by an older female who remembers distant waterholes."),
    Chunk("Octopus intelligence", "Animals & Wildlife",
          "An octopus can solve puzzles, open jars, and change color to blend into the seafloor."),
    Chunk("Wolf packs", "Animals & Wildlife",
          "Wolves hunt as a coordinated pack and use howls to keep the group together across the forest."),
    Chunk("Coral reefs", "Animals & Wildlife",
          "Coral reefs shelter thousands of fish species and are built by tiny animals over centuries."),
    Chunk("Butterfly life cycle", "Animals & Wildlife",
          "A caterpillar spins a chrysalis and transforms into a butterfly through metamorphosis."),
    Chunk("Penguins", "Animals & Wildlife",
          "Penguins huddle together for warmth and take turns shielding the group from icy winds."),
    Chunk("Spider webs", "Animals & Wildlife",
          "Spiders spin silk webs to trap insects, sensing the smallest vibration of a caught meal."),
    Chunk("Frog choruses", "Animals & Wildlife",
          "On warm wet nights frogs call together in loud choruses to attract mates near the pond."),
    Chunk("Deer in the forest", "Animals & Wildlife",
          "Deer browse quietly at dawn and dusk and freeze at any sound to avoid nearby predators."),
    Chunk("Bats at dusk", "Animals & Wildlife",
          "Bats hunt insects in the dark using echolocation, sending out clicks and listening for echoes."),
    Chunk("Sea turtles", "Animals & Wildlife",
          "Sea turtles return to the beach where they hatched to lay their eggs in the warm sand."),
    Chunk("Squirrels and acorns", "Animals & Wildlife",
          "Squirrels bury acorns across the forest and forget enough of them to grow new oak trees."),
    Chunk("Dolphins", "Animals & Wildlife",
          "Dolphins are social mammals that play, hunt cooperatively, and whistle to recognize friends."),
    Chunk("Ant colonies", "Animals & Wildlife",
          "Ants work as a colony, leaving scent trails so the others can follow the path to food."),
    Chunk("Owls at night", "Animals & Wildlife",
          "Owls have silent feathers and sharp hearing that let them hunt mice in near-total darkness."),
    Chunk("Polar bears", "Animals & Wildlife",
          "Polar bears roam the sea ice hunting seals and rely on thick fur and fat to stay warm."),
    Chunk("Hummingbirds", "Animals & Wildlife",
          "Hummingbirds beat their wings dozens of times a second and hover to sip nectar from flowers."),
    Chunk("Chameleons", "Animals & Wildlife",
          "A chameleon shifts color to signal its mood and swivels each eye to watch in two directions."),

    # Space & Astronomy
    Chunk("The solar system", "Space & Astronomy",
          "Eight planets orbit the sun, from rocky Mercury up close to icy Neptune far away."),
    Chunk("Phases of the moon", "Space & Astronomy",
          "The moon appears to change shape as sunlight lights different parts during its monthly orbit."),
    Chunk("Stars and galaxies", "Space & Astronomy",
          "A galaxy holds billions of stars, and our Milky Way is just one among countless others."),
    Chunk("Black holes", "Space & Astronomy",
          "A black hole packs so much mass into a small space that not even light can escape it."),
    Chunk("Comets", "Space & Astronomy",
          "Comets are icy travelers that grow glowing tails as they near the sun and start to melt."),
    Chunk("Volcanoes on Mars", "Space & Astronomy",
          "Mars hosts Olympus Mons, a giant volcano and the tallest known mountain in the solar system."),
    Chunk("Constellations", "Space & Astronomy",
          "Constellations are patterns of stars that cultures have used for ages to tell stories and navigate."),
    Chunk("Telescopes", "Space & Astronomy",
          "Telescopes gather faint light from distant objects so we can study planets, stars, and nebulae."),
    Chunk("The sun", "Space & Astronomy",
          "The sun is a giant ball of hot gas whose fusion gives Earth nearly all of its light and heat."),
    Chunk("Meteor showers", "Space & Astronomy",
          "Meteor showers happen when Earth passes through dust left behind by a comet's orbit."),
    Chunk("Saturn's rings", "Space & Astronomy",
          "Saturn is circled by bright rings made of countless chunks of ice and rock orbiting the planet."),
    Chunk("The International Space Station", "Space & Astronomy",
          "Astronauts live and work aboard the space station, circling Earth roughly every ninety minutes."),
    Chunk("Light years", "Space & Astronomy",
          "A light year measures distance: how far light travels in a year across the vast emptiness of space."),
    Chunk("The Big Bang", "Space & Astronomy",
          "Most evidence suggests the universe began expanding from a hot, dense state billions of years ago."),
    Chunk("Auroras", "Space & Astronomy",
          "Auroras paint the polar sky when particles from the sun collide with gases in the atmosphere."),
    Chunk("Eclipses", "Space & Astronomy",
          "A solar eclipse happens when the moon slips directly between the Earth and the sun."),
    Chunk("Asteroids", "Space & Astronomy",
          "Asteroids are rocky leftovers from the early solar system, many orbiting in a belt past Mars."),
    Chunk("Gravity in orbit", "Space & Astronomy",
          "Gravity keeps planets looping around the sun and moons circling the planets they belong to."),
    Chunk("Dwarf planet Pluto", "Space & Astronomy",
          "Pluto is a small, icy dwarf planet far out in the cold edges of the solar system."),
    Chunk("Astronaut training", "Space & Astronomy",
          "Astronauts train underwater to rehearse spacewalks and get used to floating without gravity."),

    # Food & Cooking
    Chunk("Simple tomato sauce", "Food & Cooking",
          "Simmer crushed tomatoes with garlic, olive oil, and basil for an easy weeknight pasta sauce."),
    Chunk("Baking bread", "Food & Cooking",
          "Knead flour, water, yeast, and salt, then let the dough rise before baking it in a hot oven."),
    Chunk("Knife skills", "Food & Cooking",
          "A sharp knife and a steady, curled grip make chopping vegetables faster and much safer."),
    Chunk("Seasoning to taste", "Food & Cooking",
          "Taste as you go and balance salt, acid, fat, and heat to make a dish's flavors pop."),
    Chunk("Roasting vegetables", "Food & Cooking",
          "Toss vegetables in oil and roast them hot until the edges caramelize and turn sweet."),
    Chunk("A hearty soup", "Food & Cooking",
          "Simmer beans, vegetables, and herbs in broth for a warming soup on a cold evening."),
    Chunk("Breakfast eggs", "Food & Cooking",
          "Whisk eggs and cook them low and slow with a little butter for soft, creamy scrambled eggs."),
    Chunk("Baking cookies", "Food & Cooking",
          "Cream butter and sugar, fold in flour and chocolate chips, then bake until the edges turn golden."),
    Chunk("Fresh salad", "Food & Cooking",
          "Crisp greens, a squeeze of lemon, olive oil, and a pinch of salt make a quick bright salad."),
    Chunk("Cooking rice", "Food & Cooking",
          "Rinse the rice, add the right amount of water, then cover and let it steam until fluffy."),
    Chunk("Homemade pizza", "Food & Cooking",
          "Stretch the dough, spread sauce and cheese, and bake it hot for a crisp homemade pizza."),
    Chunk("Stir-fry basics", "Food & Cooking",
          "Cook vegetables fast over high heat in a hot pan, stirring constantly so nothing burns."),
    Chunk("Brewing coffee", "Food & Cooking",
          "Use fresh grounds and water just off the boil to brew a balanced, flavorful cup of coffee."),
    Chunk("Pancakes", "Food & Cooking",
          "Mix batter until just combined and pour onto a warm griddle until bubbles form, then flip."),
    Chunk("Smoothies", "Food & Cooking",
          "Blend fruit, yogurt, and a handful of greens for a quick smoothie packed with nutrients."),
    Chunk("Spices and herbs", "Food & Cooking",
          "A pinch of cumin, paprika, or fresh herbs can turn a plain dish into something memorable."),
    Chunk("Meal prep", "Food & Cooking",
          "Cooking a few meals ahead on the weekend saves time and money during a busy week."),
    Chunk("Grilling outdoors", "Food & Cooking",
          "Let the grill heat fully, oil the grate, and flip food once for clean, smoky grill marks."),
    Chunk("Baking a cake", "Food & Cooking",
          "Measure ingredients carefully and don't overmix so the cake bakes up light and tender."),
    Chunk("Leftovers", "Food & Cooking",
          "Storing leftovers in sealed containers keeps food fresh and makes tomorrow's lunch easy."),

    # Weather & Climate
    Chunk("How rain forms", "Weather & Climate",
          "Water evaporates, cools into clouds, and falls back as rain when the droplets grow heavy."),
    Chunk("Thunderstorms", "Weather & Climate",
          "Thunderstorms build when warm humid air rises fast, sparking lightning and rumbling thunder."),
    Chunk("The water cycle", "Weather & Climate",
          "Water moves endlessly between oceans, sky, and land through evaporation and precipitation."),
    Chunk("Snowflakes", "Weather & Climate",
          "Snowflakes form as water vapor freezes into delicate six-sided crystals high in cold clouds."),
    Chunk("Hurricanes", "Weather & Climate",
          "Hurricanes are huge spinning storms that gather strength over warm tropical ocean water."),
    Chunk("Rainbows", "Weather & Climate",
          "A rainbow appears when sunlight bends and splits inside raindrops into a band of colors."),
    Chunk("Fog", "Weather & Climate",
          "Fog is simply a cloud at ground level, formed when moist air cools near the surface."),
    Chunk("The seasons", "Weather & Climate",
          "Earth's tilt gives us seasons as different parts lean toward or away from the sun each year."),
    Chunk("Wind", "Weather & Climate",
          "Wind is moving air that flows from areas of high pressure toward areas of lower pressure."),
    Chunk("Droughts", "Weather & Climate",
          "A drought is a long stretch of dry weather that lowers rivers and stresses crops and soil."),
    Chunk("Humidity", "Weather & Climate",
          "Humidity measures how much water vapor is in the air, and high humidity makes heat feel worse."),
    Chunk("Tornadoes", "Weather & Climate",
          "A tornado is a violent funnel of spinning wind that drops from a powerful storm cloud."),
    Chunk("Climate vs weather", "Weather & Climate",
          "Weather is what happens today, while climate is the average pattern over many years."),
    Chunk("Heat waves", "Weather & Climate",
          "A heat wave is a stretch of unusually hot days that can be dangerous without shade and water."),
    Chunk("Clouds", "Weather & Climate",
          "Clouds come in many shapes, from wispy high cirrus to towering storm-bringing cumulonimbus."),
    Chunk("Frost", "Weather & Climate",
          "Frost forms on cold clear nights when surfaces drop below freezing and dew turns to ice."),
    Chunk("Weather forecasting", "Weather & Climate",
          "Forecasters use satellites and models to predict tomorrow's temperature, rain, and wind."),
    Chunk("Monsoons", "Weather & Climate",
          "Monsoon winds shift with the seasons, bringing heavy rains that many regions depend on."),
    Chunk("The greenhouse effect", "Weather & Climate",
          "Certain gases trap heat in the atmosphere, keeping the planet warm enough to support life."),
    Chunk("Hailstorms", "Weather & Climate",
          "Hail forms when strong updrafts toss raindrops high enough to freeze into icy balls."),

    # Music & Art
    Chunk("Learning guitar", "Music & Art",
          "A beginner guitarist starts with a few chords and slow strumming before playing full songs."),
    Chunk("Reading sheet music", "Music & Art",
          "Sheet music uses notes on a staff to show which pitches to play and how long to hold them."),
    Chunk("Color theory", "Music & Art",
          "Color theory explains how hues combine, with complementary colors sitting opposite on the wheel."),
    Chunk("Sketching basics", "Music & Art",
          "Start a sketch with light, loose shapes, then darken the lines once the proportions look right."),
    Chunk("Rhythm and beat", "Music & Art",
          "Rhythm is the pattern of sounds in time, and a steady beat keeps musicians playing together."),
    Chunk("Watercolor painting", "Music & Art",
          "Watercolor flows best on wet paper, blending soft washes of color that dry lighter than they look."),
    Chunk("The orchestra", "Music & Art",
          "An orchestra blends strings, woodwinds, brass, and percussion under a conductor's lead."),
    Chunk("Singing in tune", "Music & Art",
          "Singing in tune comes from listening closely and matching your voice to the right pitch."),
    Chunk("Photography composition", "Music & Art",
          "Placing your subject off-center along thirds often makes a photograph feel more balanced."),
    Chunk("Pottery", "Music & Art",
          "A potter centers wet clay on a spinning wheel and shapes it into bowls with steady hands."),
    Chunk("Major and minor keys", "Music & Art",
          "Major keys tend to sound bright and happy, while minor keys feel darker or more thoughtful."),
    Chunk("Street murals", "Music & Art",
          "Large murals turn blank city walls into bold public art that anyone passing by can enjoy."),
    Chunk("Playing piano", "Music & Art",
          "Pianists train both hands to move independently, the left holding rhythm and the right the melody."),
    Chunk("Sculpture", "Music & Art",
          "A sculptor carves, molds, or assembles material into a form you can walk around and view."),
    Chunk("Songwriting", "Music & Art",
          "Songwriters pair a memorable melody with lyrics and a chord progression to tell a short story."),
    Chunk("Drawing perspective", "Music & Art",
          "Lines that meet at a vanishing point give a flat drawing a convincing sense of depth."),
    Chunk("Drums and percussion", "Music & Art",
          "Drummers keep the tempo and drive a song forward with steady, repeating rhythmic patterns."),
    Chunk("Museums of art", "Music & Art",
          "An art museum gathers paintings and sculptures so visitors can see works from many eras up close."),
    Chunk("Mixing paint", "Music & Art",
          "Mixing the three primary colors in different amounts can produce nearly any shade you need."),
    Chunk("Jazz improvisation", "Music & Art",
          "Jazz musicians improvise, inventing new melodies on the spot over a shared set of chords."),

    # Books & Stories
    Chunk("Building a plot", "Books & Stories",
          "A classic plot follows a character who wants something, faces obstacles, and changes by the end."),
    Chunk("Memorable characters", "Books & Stories",
          "Strong characters feel real because they have clear desires, flaws, and a voice of their own."),
    Chunk("Setting the scene", "Books & Stories",
          "A vivid setting grounds a story, letting readers picture the time and place the action unfolds."),
    Chunk("Fairy tales", "Books & Stories",
          "Fairy tales pass down simple lessons through talking animals, clever heroes, and magic spells."),
    Chunk("Mystery novels", "Books & Stories",
          "A good mystery scatters clues and red herrings so readers race the detective to the answer."),
    Chunk("Poetry", "Books & Stories",
          "Poetry packs feeling into few words, using rhythm, imagery, and sometimes rhyme to move readers."),
    Chunk("Libraries", "Books & Stories",
          "A library lends books for free and offers a quiet place to read, study, and discover new authors."),
    Chunk("Science fiction", "Books & Stories",
          "Science fiction imagines future technology and asks how it might change the way people live."),
    Chunk("The narrator's voice", "Books & Stories",
          "Who tells the story shapes everything, from a trusted guide to an unreliable first-person voice."),
    Chunk("Plot twists", "Books & Stories",
          "A satisfying twist surprises readers yet feels earned once they look back at the early hints."),
    Chunk("Writing dialogue", "Books & Stories",
          "Good dialogue sounds natural, reveals character, and pushes the scene forward at the same time."),
    Chunk("Folktales", "Books & Stories",
          "Folktales travel by word of mouth across generations, shifting a little with every retelling."),
    Chunk("Biographies", "Books & Stories",
          "A biography tells the true story of a real person's life, struggles, and lasting achievements."),
    Chunk("Reading habits", "Books & Stories",
          "Setting aside a few quiet minutes each day makes finishing even a long book feel easy."),
    Chunk("Fantasy worlds", "Books & Stories",
          "Fantasy authors invent whole worlds with their own maps, history, creatures, and rules of magic."),
    Chunk("Short stories", "Books & Stories",
          "A short story captures a single moment or idea, often with one sharp turn near the end."),
    Chunk("Editing a draft", "Books & Stories",
          "Revising means cutting weak lines and tightening sentences until every word earns its place."),
    Chunk("Book clubs", "Books & Stories",
          "A book club gathers readers to share opinions and notice details one person might miss alone."),
    Chunk("Comics and graphic novels", "Books & Stories",
          "Comics tell stories through panels, blending art and a few words to show action and emotion."),
    Chunk("Adventure tales", "Books & Stories",
          "Adventure stories send heroes on a journey full of danger, discovery, and hard-won growth."),

    # Cities & Travel
    Chunk("Packing light", "Cities & Travel",
          "Packing light with a single carry-on saves time at the airport and keeps a trip simple."),
    Chunk("City museums", "Cities & Travel",
          "Many cities have museums where you can spend an afternoon among art, history, and science."),
    Chunk("Public transit", "Cities & Travel",
          "Buses, subways, and trams make it cheap and easy to reach almost any corner of a big city."),
    Chunk("Trying local food", "Cities & Travel",
          "Eating at small neighborhood spots is one of the best ways to get to know a new place."),
    Chunk("Walking tours", "Cities & Travel",
          "A walking tour reveals hidden alleys, street art, and stories you would miss from a bus."),
    Chunk("Famous landmarks", "Cities & Travel",
          "Towers, bridges, and old squares give a city its skyline and draw visitors from around the world."),
    Chunk("Booking a hostel", "Cities & Travel",
          "Hostels offer cheap beds and a social common room where travelers swap tips and make friends."),
    Chunk("Scenic train rides", "Cities & Travel",
          "A train lets you watch mountains and coastlines roll past on the way between distant cities."),
    Chunk("Reading a map", "Cities & Travel",
          "A paper or phone map helps you find your way and spot parks, stations, and sights nearby."),
    Chunk("Street markets", "Cities & Travel",
          "Open-air markets bustle with fresh produce, crafts, and the everyday life of a neighborhood."),
    Chunk("Old town centers", "Cities & Travel",
          "Historic old towns have narrow cobbled streets, cafes, and buildings hundreds of years old."),
    Chunk("Budget travel", "Cities & Travel",
          "Traveling on a budget means cooking some meals, walking more, and choosing free attractions."),
    Chunk("Coastal towns", "Cities & Travel",
          "Seaside towns offer harbor views, fresh seafood, and a slower pace than a busy capital."),
    Chunk("Parks and gardens", "Cities & Travel",
          "City parks give residents and visitors green space to rest, picnic, and watch the world go by."),
    Chunk("Travel planning", "Cities & Travel",
          "Listing a few must-see spots but leaving free time lets you wander and find happy surprises."),
    Chunk("Cathedrals and temples", "Cities & Travel",
          "Grand places of worship show off centuries of architecture, art, and quiet open space."),
    Chunk("Airports", "Cities & Travel",
          "Arriving early eases the stress of check-in, security lines, and finding the right gate."),
    Chunk("Skylines at night", "Cities & Travel",
          "A city's lit skyline looks magical from a rooftop, riverbank, or nearby hilltop after dark."),
    Chunk("Local festivals", "Cities & Travel",
          "Catching a local festival lets travelers join the music, food, and traditions of a region."),
    Chunk("Day trips", "Cities & Travel",
          "A short day trip to a nearby village or beach adds variety without changing your hotel."),

    # Health & Wellness
    Chunk("Drinking water", "Health & Wellness",
          "Sipping water through the day keeps you hydrated, steadies energy, and supports clear thinking."),
    Chunk("A balanced plate", "Health & Wellness",
          "Filling half your plate with vegetables and adding protein and whole grains keeps meals balanced."),
    Chunk("Getting good sleep", "Health & Wellness",
          "A regular bedtime and a dark, cool room help you fall asleep and wake up feeling rested."),
    Chunk("Daily walks", "Health & Wellness",
          "A brisk daily walk lifts your mood, helps your heart, and clears a cluttered mind."),
    Chunk("Stretching", "Health & Wellness",
          "Gentle stretching keeps joints limber, eases stiffness, and feels great after sitting all day."),
    Chunk("Managing stress", "Health & Wellness",
          "Slow breathing, short breaks, and time outdoors can ease stress on a hectic day."),
    Chunk("Healthy breakfast", "Health & Wellness",
          "A breakfast with protein and fruit gives steady energy instead of a quick sugar crash."),
    Chunk("Washing hands", "Health & Wellness",
          "Washing hands with soap for twenty seconds is a simple way to stop germs from spreading."),
    Chunk("Mindful breathing", "Health & Wellness",
          "Pausing to take a few slow, deep breaths can calm the body and quiet a busy mind."),
    Chunk("Posture at a desk", "Health & Wellness",
          "Sitting tall with feet flat and the screen at eye level eases strain on your neck and back."),
    Chunk("Limiting screen time", "Health & Wellness",
          "Setting the phone aside before bed helps your eyes rest and your brain wind down for sleep."),
    Chunk("Staying active", "Health & Wellness",
          "Small bursts of movement, like taking the stairs, add up to a more active and healthier day."),
    Chunk("Eating more vegetables", "Health & Wellness",
          "Adding a serving of vegetables to each meal boosts fiber, vitamins, and lasting fullness."),
    Chunk("Routine checkups", "Health & Wellness",
          "Regular checkups catch small health issues early, when they are easiest to treat."),
    Chunk("Sun protection", "Health & Wellness",
          "Sunscreen, a hat, and shade protect your skin from sunburn on bright summer days."),
    Chunk("Hand stretches", "Health & Wellness",
          "Brief hand and wrist stretches ease tension for anyone who types or writes for hours."),
    Chunk("Listening to your body", "Health & Wellness",
          "Resting when you feel run down helps you recover faster than pushing through exhaustion."),
    Chunk("Friendship and health", "Health & Wellness",
          "Staying connected with friends and family supports both mood and long-term well-being."),
    Chunk("Cutting back on sugar", "Health & Wellness",
          "Swapping sugary drinks for water steadies energy and is gentler on your teeth and waistline."),
    Chunk("A calming routine", "Health & Wellness",
          "A simple evening routine signals your body it is time to relax and prepare for sleep."),

    # Technology Basics
    Chunk("What is the internet", "Technology Basics",
          "The internet is a giant network that lets computers around the world share information."),
    Chunk("How email works", "Technology Basics",
          "Email sends a message from your account through servers to land in someone else's inbox."),
    Chunk("Strong passwords", "Technology Basics",
          "A long passphrase that mixes words and symbols is far harder to guess than a short one."),
    Chunk("Saving files", "Technology Basics",
          "Saving your work often, and keeping a backup copy, protects it if a device suddenly fails."),
    Chunk("Wi-Fi networks", "Technology Basics",
          "Wi-Fi connects your devices to the internet without cables using radio signals from a router."),
    Chunk("Apps and software", "Technology Basics",
          "An app is a program built to do one job, like sending messages or editing a photo."),
    Chunk("Cloud storage", "Technology Basics",
          "Cloud storage keeps your files on remote servers so you can reach them from any device."),
    Chunk("Software updates", "Technology Basics",
          "Installing updates fixes bugs and patches security holes that attackers might otherwise use."),
    Chunk("Web browsers", "Technology Basics",
          "A browser loads websites by fetching pages and showing their text, images, and links."),
    Chunk("Keyboard shortcuts", "Technology Basics",
          "Learning a few keyboard shortcuts, like copy and paste, speeds up everyday computer tasks."),
    Chunk("Avoiding scams", "Technology Basics",
          "Be wary of emails that rush you to click a link or share a password, as these are often scams."),
    Chunk("Bits and bytes", "Technology Basics",
          "Computers store everything as bits, tiny ones and zeros that combine to form text and images."),
    Chunk("Smartphones", "Technology Basics",
          "A smartphone is a pocket computer that handles calls, photos, maps, and countless apps."),
    Chunk("Backing up data", "Technology Basics",
          "Keeping a second copy of important files on a drive or the cloud guards against accidents."),
    Chunk("Search engines", "Technology Basics",
          "A search engine scans the web and ranks pages so you can quickly find what you need."),
    Chunk("Two-factor login", "Technology Basics",
          "Adding a second step, like a code on your phone, makes it much harder to break into accounts."),
    Chunk("Hardware vs software", "Technology Basics",
          "Hardware is the physical machine you can touch, while software is the instructions it runs."),
    Chunk("Charging batteries", "Technology Basics",
          "Avoiding extreme heat and very low charges helps a device battery last longer over the years."),
    Chunk("Operating systems", "Technology Basics",
          "An operating system manages a device's hardware and lets other programs run on top of it."),
    Chunk("Spam filters", "Technology Basics",
          "Spam filters sort junk mail out of your inbox by spotting the patterns scammers tend to use."),

    # Sports & Recreation
    Chunk("Soccer basics", "Sports & Recreation",
          "Soccer players move the ball with their feet, passing and dribbling to score in the goal."),
    Chunk("Swimming laps", "Sports & Recreation",
          "Swimming works the whole body, and steady breathing helps you glide smoothly through the water."),
    Chunk("Hiking trails", "Sports & Recreation",
          "A good pair of shoes and plenty of water make a long hike up a mountain trail far more enjoyable."),
    Chunk("Playing basketball", "Sports & Recreation",
          "Basketball blends dribbling, passing, and shooting as two teams race up and down the court."),
    Chunk("Cycling", "Sports & Recreation",
          "Riding a bike builds leg strength and is a fun, low-impact way to explore your neighborhood."),
    Chunk("Yoga", "Sports & Recreation",
          "Yoga links gentle poses with steady breathing to build flexibility, balance, and calm."),
    Chunk("Tennis", "Sports & Recreation",
          "Tennis players rally back and forth across a net, mixing power and placement to win points."),
    Chunk("Camping", "Sports & Recreation",
          "Pitching a tent and cooking over a fire makes a night under the stars feel like an adventure."),
    Chunk("Running a 5K", "Sports & Recreation",
          "Training for a 5K means slowly building distance until running the full route feels comfortable."),
    Chunk("Rock climbing", "Sports & Recreation",
          "Climbers read the wall for handholds and trust their grip, balance, and a safety rope."),
    Chunk("Fishing", "Sports & Recreation",
          "Fishing rewards patience as you cast a line, wait for a bite, and enjoy the quiet of the water."),
    Chunk("Table tennis", "Sports & Recreation",
          "Table tennis is fast and fun, with quick reflexes and spin deciding most of the rallies."),
    Chunk("Volleyball", "Sports & Recreation",
          "Volleyball teams bump, set, and spike to keep the ball off their side and over the net."),
    Chunk("Skateboarding", "Sports & Recreation",
          "Skateboarders practice balance and timing for hours to land even a simple trick cleanly."),
    Chunk("Kayaking", "Sports & Recreation",
          "Paddling a kayak along a calm river or lake is a peaceful way to spend a sunny afternoon."),
    Chunk("Golf", "Sports & Recreation",
          "Golf rewards a smooth, repeatable swing as players try to reach each hole in the fewest shots."),
    Chunk("Stretching before sport", "Sports & Recreation",
          "A light warm-up and easy stretches loosen muscles and lower the chance of a pulled muscle."),
    Chunk("Bowling", "Sports & Recreation",
          "Bowling is a relaxed game where you roll a ball down a lane and aim to knock down all the pins."),
    Chunk("Frisbee in the park", "Sports & Recreation",
          "Tossing a frisbee is an easy, free way to get outside and move around with friends."),
    Chunk("Ice skating", "Sports & Recreation",
          "Ice skating takes a little balance at first, but gliding across the rink soon feels effortless."),
]


# ---------------------------------------------------------------------------
# Modeling: TF-IDF vectors + TruncatedSVD projection to 3D. Cached so the math
# only runs once per session.
# ---------------------------------------------------------------------------


@st.cache_resource
def build_index():
    """Fit the vectorizer and 3D projection over the sample chunks."""
    texts = [f"{c.title}. {c.text}" for c in CHUNKS]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    # Real embeddings live in many dimensions; we project down to 3 so humans can
    # see the semantic space. TruncatedSVD works directly on sparse TF-IDF
    # matrices (PCA does not).
    svd = TruncatedSVD(n_components=3, random_state=42)
    coords = svd.fit_transform(tfidf)

    df = pd.DataFrame(
        {
            "title": [c.title for c in CHUNKS],
            "category": [c.category for c in CHUNKS],
            "text": [c.text for c in CHUNKS],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
        }
    )
    return vectorizer, svd, tfidf, df


def wrap(text: str, width: int = 40) -> str:
    """Insert <br> tags so long hover text wraps nicely in Plotly."""
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB color to an rgba() string.

    Scatter3d markers only accept a scalar opacity, so to fade individual points
    we bake the alpha into a per-point color instead.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vector Database Visualized", page_icon="🧭", layout="wide")

st.title("Vector Database Visualized")
st.markdown(
    "A tiny interactive demo that turns text chunks into vectors so you can "
    "**see how semantic search works.**"
)

# Sidebar: plain-English explanations.
with st.sidebar:
    st.header("How this works")
    st.markdown(
        """
**What is an embedding?**
An embedding turns a piece of text into a list of numbers (a *vector*). Texts
with similar meaning get similar numbers.

**What is a vector database?**
A store of those vectors that can quickly find which ones are closest to a
query vector, that's how it "searches by meaning" instead of exact keywords.

**What is nearest-neighbor search?**
Given your query's vector, it finds the stored vectors with the smallest
distance (here, highest cosine similarity) — the *nearest neighbors*.

**Why do similar ideas appear near each other?**
Because similar text produces similar vectors, related chunks land close
together in space. The clusters you see are themes grouping themselves —
**points closer together are more similar in meaning.**

**Wait, are vectors really 3D?**
No. Real vector databases usually store **high-dimensional** vectors (hundreds
or thousands of numbers each). That's impossible to picture, so this app
**projects them down into 3D** purely so humans can see the semantic space.
        """
    )
    st.caption(
        "Local educational demo using TF-IDF + TruncatedSVD (projected to 3D). "
        "Not a production vector database."
    )

vectorizer, svd, tfidf, df = build_index()

# Controls.
col_query, col_k = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Search the vector database",
        placeholder="e.g. planets and stars, healthy breakfast, forest animals…",
    )
with col_k:
    top_k = st.slider("Top-k", min_value=1, max_value=12, value=5)

st.caption(
    "Try an example: *planets and stars · healthy breakfast · forest animals · "
    "rainy weather · city museums · beginner guitar · volcanoes and mountains*"
)

# Compute similarity if there's a query.
plot_df = df.copy()
plot_df["similarity"] = np.nan
top_idx: list[int] = []
query_xyz = None

if query.strip():
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf)[0]
    plot_df["similarity"] = sims
    top_idx = list(np.argsort(sims)[::-1][:top_k])
    # Project the query into the same 3D space as the stored chunks.
    query_xyz = svd.transform(q_vec)[0]

# ---------------------------------------------------------------------------
# 3D scatter plot. Rotate, drag, and zoom to explore the semantic space.
# ---------------------------------------------------------------------------

fig = go.Figure()

categories = sorted(plot_df["category"].unique())
# A high-contrast, saturated palette so each of the ten categories is easy to
# tell apart against the light 3D background.
palette = [
    "#E6194B",  # red
    "#3CB44B",  # green
    "#0082C8",  # blue
    "#F58231",  # orange
    "#911EB4",  # purple
    "#00C8C8",  # teal
    "#F032E6",  # magenta
    "#9A6324",  # brown
    "#808000",  # olive
    "#46509B",  # indigo
]
color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

has_query = bool(query.strip())
is_match = [i in set(top_idx) for i in range(len(plot_df))]

for cat in categories:
    cat_mask = plot_df["category"] == cat
    sub = plot_df[cat_mask]
    sub_match = [is_match[i] for i in sub.index]

    customdata = np.stack(
        [
            sub["title"],
            sub["category"],
            [wrap(t) for t in sub["text"]],
            [f"{s:.3f}" if not np.isnan(s) else "—" for s in sub["similarity"]],
        ],
        axis=-1,
    )

    # When a query is active, emphasize the top-k matches and fade/shrink the
    # rest. Scatter3d takes per-point size and color arrays (but only a scalar
    # opacity), so we fade non-matches via per-point rgba colors.
    base = color_map[cat]
    fig.add_trace(
        go.Scatter3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["z"],
            mode="markers",
            name=cat,
            marker=dict(
                size=[
                    16 if m else (6 if has_query else 9)
                    for m in sub_match
                ],
                color=[
                    hex_to_rgba(base, 1.0 if (m or not has_query) else 0.25)
                    for m in sub_match
                ],
                line=dict(
                    width=0 if has_query else 1.0,
                    color="rgba(255, 255, 255, 0.9)",
                ),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<i>%{customdata[1]}</i><br><br>"
                "%{customdata[2]}<br><br>"
                "similarity: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

# Mark the query point with a visually distinct marker.
if query_xyz is not None:
    fig.add_trace(
        go.Scatter3d(
            x=[query_xyz[0]],
            y=[query_xyz[1]],
            z=[query_xyz[2]],
            mode="markers+text",
            name="Your query",
            text=["query"],
            textposition="top center",
            marker=dict(size=16, color="#111111", symbol="diamond",
                        line=dict(width=3, color="#FFD500")),
            hovertemplate=f"<b>Your query</b><br>{wrap(query)}<extra></extra>",
        )
    )

axis_style = dict(
    showbackground=True,
    backgroundcolor="rgba(248, 249, 251, 0.9)",
    gridcolor="#C9CED6",
    zeroline=False,
)
fig.update_layout(
    height=860,
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02, xanchor="left", x=0,
        itemsizing="constant",
        font=dict(size=13),
    ),
    hoverlabel=dict(font_size=13, font_family="sans-serif", align="left"),
    scene=dict(
        xaxis=dict(title="dim 1", **axis_style),
        yaxis=dict(title="dim 2", **axis_style),
        zaxis=dict(title="dim 3", **axis_style),
        aspectmode="cube",
    ),
)

st.plotly_chart(fig, width="stretch")
st.caption("Tip: drag to rotate, scroll to zoom, and hover any point for details.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

if query.strip():
    st.subheader(f"Top {top_k} nearest chunks")
    results = (
        plot_df.loc[top_idx, ["title", "category", "similarity", "text"]]
        .rename(
            columns={
                "title": "Title",
                "category": "Category",
                "similarity": "Similarity",
                "text": "Text",
            }
        )
        .reset_index(drop=True)
    )
    results.index = results.index + 1  # rank starting at 1
    st.dataframe(
        results.style.format({"Similarity": "{:.3f}"}),
        width="stretch",
    )
    if results["Similarity"].max() <= 0:
        st.info(
            "No overlap with the stored chunks. Try wording your query with "
            "topics like space, animals, food, weather, music, or sports."
        )
else:
    st.info("Type a query above to run a semantic search and highlight the nearest chunks.")
