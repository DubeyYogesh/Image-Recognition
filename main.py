from ultralytics import YOLO
import cvzone
import cv2
import math

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# model = YOLO('yolov8n.yaml')     #build a new model from YAML
model = YOLO('yolov8l.pt')             # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')         # build from YAML and transfer weights

# Train the model
# results = model.train(data='smartwatch.yaml', epochs=10)

results = model("Images/4.jpg", show=True)
cv2.waitKey(0)

className = ['Person', 'Bicycle', 'Television', 'Refrigerator', 'Laptop', 'Chair', 'Table', 'Cellphone', 'Book', 'Clock', 'Sunglasses', 'Wallet', 'Headphones', 'Car', 'Backpack', 'Shoes', 'Camera', 'Hat', 'Watch', 'Water Bottle', 'Guitar', 'Pen', 'Keychain', 'Bicycle Helmet', 'Baseball Bat', 'Soccer Ball', 'Tennis Racket', 'Basketball', 'Football', 'Tennis Ball', 'Volleyball', 'Golf Club', 'Fishing Rod', 'Surfboard', 'Skateboard', 'Snowboard', 'Canoe', 'Kayak', 'Paddle', 'Tennis Shoes', 'Running Shoes', 'Hiking Boots', 'Soccer Cleats', 'Swimsuit', 'Goggles', 'Snorkel', 'Flippers', 'Life Jacket', 'Yoga Mat', 'Dumbbells', 'Jump Rope', 'Treadmill', 'Yoga Blocks', 'Resistance Bands', 'Punching Bag', 'Boxing Gloves', 'Tennis Net', 'Golf Ball', 'Golf Tee', 'Fishing Lures', 'Camping Tent', 'Sleeping Bag', 'Campfire', 'Hiking Map', 'Compass', 'Binoculars', 'Bird Watching Guidebook', 'Climbing Rope', 'Carabiner', 'Helmet', 'Harness', 'Belay Device', 'Chalk Bag', 'Climbing Shoes', 'Rock Climbing Holds', 'Belay Glasses', 'Chalk Ball', 'Climbing Harness', 'Rope Bag', 'Belay Gloves', 'Quickdraws', 'Swimming Pool', 'Swim Cap', 'Swim Goggles', 'Swim Fins', 'Swim Paddles', 'Swim Snorkel', 'Swim Earplugs', 'Swim Nose Clip', 'Swim Kickboard', 'Swim Pull Buoy', 'Swim Mesh Bag', 'Swim Towel', 'Swim Water Bottle', 'Swim Timer', 'Swim Lap Counter', 'Swim Meet Backstroke Flags', 'Swim Pool Lane Ropes', 'Swim Starting Blocks', 'Swim Touch Pads', 'Swim Meet Scoring Table', 'Swim Meet Awards', 'Swim Meet Ribbons', 'Swim Meet Trophies', 'Swim Meet Medals', 'Swim Meet Championship Banner', 'Badminton Net', 'Badminton Racket', 'Badminton Shuttlecock', 'Badminton Court', 'Badminton Shoes', 'Badminton Bag', 'Badminton Socks', 'Badminton Grip', 'Badminton String', 'Badminton Overgrip', 'Badminton Hat', 'Badminton Towel', 'Badminton Shirt', 'Badminton Skirt', 'Badminton Shorts', 'Baseball Glove', 'Baseball Ball', 'Baseball Cap', 'Baseball Jersey', 'Baseball Pants', 'Baseball Socks', 'Baseball Cleats', 'Baseball Helmet', 'Baseball Chest Protector', 'Baseball Leg Guards', "Baseball Catcher's Mitt", 'Baseball Umpire Mask', 'Baseball Umpire Chest Protector', 'Baseball Umpire Leg Guards', 'Baseball Umpire Indicator', 'Baseball Umpire Plate Brush', 'Baseball Umpire Ball Bag', 'Baseball Umpire Plate Shoes', 'Baseball Umpire Shin Guards', 'Baseball Pitching Machine', 'Baseball Scorebook', 'Baseball Base Set', "Baseball Pitcher's Rubber", 'Baseball Batting Tee', 'Baseball Hitting Net', 'Baseball Batting Gloves', 'Baseball Bat Weight', "Baseball Pitcher's Mound", 'Baseball Training Balls', 'Baseball Throwing Target', 'Basketball Hoop', 'Basketball Court', 'Basketball Jersey', 'Basketball Shoes', 'Basketball Net', 'Basketball Pump', 'Basketball Shorts', 'Basketball Socks', 'Basketball Whistle', 'Basketball Referee Shirt', 'Basketball Referee Whistle', 'Basketball Backboard', 'Basketball Scoreboard', 'Basketball Shot Clock', 'Basketball Cone', 'Basketball Knee Pads', 'Basketball Elbow Pads', 'Basketball Ankle Brace', 'Basketball Headband', 'Basketball Wristbands', 'Basketball Referee Hat', 'Basketball Referee Shoes', 'Basketball Referee Pants', 'Basketball Umpire Mask', 'Basketball Umpire Chest Protector', 'Basketball Umpire Leg Guards', 'Basketball Umpire Indicator', 'Basketball Umpire Plate Brush', 'Basketball Umpire Ball Bag', 'Basketball Umpire Plate Shoes', 'Basketball Umpire Shin Guards', 'Basketball Pitching Machine', 'Basketball Scorebook', 'Basketball Base Set', "Basketball Pitcher's Rubber", 'Basketball Batting Tee', 'Basketball Hitting Net', 'Basketball Batting Gloves', 'Basketball Bat Weight', "Basketball Pitcher's Mound", 'Basketball Training Balls', 'Basketball Throwing Target', 'Tennis Court', 'Tennis Bag', 'Tennis Socks', 'Tennis Grip', 'Tennis String', 'Tennis Overgrip', 'Tennis Hat', 'Tennis Towel', 'Tennis Shirt', 'Tennis Skirt', 'Tennis Shorts', 'Tennis Wristband', 'Volleyball Net', 'Volleyball Shoes', 'Volleyball Bag', 'Volleyball Socks', 'Volleyball Knee Pads', 'Volleyball Elbow Pads', 'Volleyball Ankle Brace', 'Volleyball Headband', 'Volleyball Wristbands', 'Volleyball Referee Hat', 'Volleyball Referee Shoes', 'Volleyball Referee Pants', 'Volleyball Referee Whistle', 'Volleyball Backboard', 'Volleyball Scoreboard', 'Volleyball Cone', 'Golf Shoes', 'Golf Gloves', 'Golf Hat', 'Golf Towel', 'Golf Shirt', 'Golf Shorts', 'Golf Socks', 'Golf Cart', 'Golf Scorecard', 'Golf Ball Marker', 'Golf Divot Tool', 'Fishing Hooks', 'Fishing Tackle Box', 'Fishing Hat', 'Fishing Vest', 'Fishing Boots', 'Fishing Gloves', 'Fishing Sunglasses', 'Fishing Rod Holder', 'Surf Leash', 'Surf Traction Pad', 'Surfboard Bag', 'Surfboard Rack', 'Surfboard Repair Kit', 'Skateboard Trucks', 'Skateboard Wheels', 'Skateboard Bearings', 'Skateboard Grip Tape', 'Skateboard Helmet', 'Skateboard Pads', 'Skateboard Shoes', 'Skateboard Backpack', 'Skateboard Tool', 'Snowboard Bindings', 'Snowboard Jacket', 'Snowboard Pants', 'Snowboard Goggles', 'Snowboard Helmet', 'Snowboard Gloves', 'Snowboard Socks', 'Snowboard Wax', 'Kayak Paddle', 'Canoe Paddle', 'Kayak Life Jacket', 'Canoe Life Jacket', 'Paddle Leash', 'Yoga Strap', 'Yoga Towel', 'Yoga Bag', 'Yoga Pants', 'Yoga Leggings', 'Yoga Top', 'Yoga Bra', 'Weight Bench', 'Weight Plates', 'Kettlebell', 'Exercise Ball', 'Exercise Mat', 'Exercise Bike', 'Rowing Machine', 'Ab Roller', 'Boxing Speed Bag', 'Boxing Heavy Bag', 'Boxing Ring', 'Boxing Shoes', 'Boxing Shorts', 'Boxing Headgear', 'Boxing Mouthguard', 'Volleyball Court', 'Golf Bag', 'Fishing Reel', 'Skateboard Deck', 'Snowboard Boots']

# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # With opencv
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(img, (x1, y1, x2, y2), (255, 0, 255), 3)
#             print(x1, y1, x2, y2)
#
#             # with cvzone
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#
#             # confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img, f'{results} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)