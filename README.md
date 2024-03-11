CHAPTER 1
INTRODUCTION


The Evolution of the world is more advanced in Technology as well as in Population. Control of the growth of the population is very hard, but we use technology to control the whole population to gather in one place. We humans have a psychological factor to celebrate or spend a good time with loved ones which might be two or three or many more. There will be a major problem if the number of individuals increases at the same place. The consequences of the power of the crowd will lead to destruction [1] like a family of 15 people being at the same place and another family with the same strength coming to meet each other, this may lead to major safety factors like building Infrastructure, ventilations, building capability, fire exits, etc.
     Large gatherings always have a vital role in the life of people, but that is not at present movement. One of the key characteristics of the modern era is the replacement of individual conscious activity with the unconscious action of crowds [1]. Crowd catastrophes, in which people suffer severe injuries or lose lives as a result of being crushed or tramped on, are not only caused by emergencies like fires, violent crowd conditions, or the excessive elation of some crowd members. Such incidents can happen everywhere, including sporting events, religious ceremonies, and rock concerts. [2]. Disasters caused by crowding have been defined by Fruin as a specific type of pedestrian traffic process in which certain 
critical performance limits have been exceeded. Fruin defines crowding as the sudden gathering of a large number of people in an enclosed space with sufficient mass and force to cause human injury or death even with the latest common problem of Covid pandamic[3].
     Proper arrangements should be arranged or backups have to be kept to control the catastrophe cases of the crowd disasters. Here are some of the records, on 11 October 1711, a collision between a carriage and a cart led to trap a large crowd in the middle of the Guillotière bridge in Lyon, France, which led to the death of 245 people. On May 30, 1770, a firework display commemorating the union of the future King Louis XVI and his consort, Marie Antoinette, in the Place de la Concord in Paris, France, led to the death of at least 133 people. The fire was caused by a malfunctioning mannequin and other decorations, leading to a panic in which many onlookers were crushed beneath their feet and some were drowned in the nearby river. The 1823 Carnival tragedy in Malta, during the celebration of Carnival, resulted in the death of 110 young boys attempting to leave the Brooklyn Theatre after attending a concert featuring the musical group Minori Oservanti. As a result of the tragedy, 278 people lost their lives on December 5, 1876.
     Brooklyn Theatre fire occurred on 18 May 1896 where 1389 people died during the coronation of Tsar Nicholas II because of crowd crush. 71 died on 4 March 2010 in Pratapgarh stampede (India), The Phnom Penh (Cambodia) stampede killed 347 people on November 22, 2010, 102 died on 15 January 2011 in 2011 Sabarimala crowd crush (India), On February 1st, 2012, there was a huge disturbance at the Port Said Stadium in Egypt, resulting in the death of 74 people, 242 people lost their lives in the Kiss nightclub fire in Brazil on January 27, 2013. Another 26 people were injured in a stampede that occurred shortly after Dussehra at Gandhi Maidan in Patna, India. 135 died in the Kanjuruhan Stadium disaster, Indonesia on 1 October 2022. Halloween revelers suffered injuries in a small lane in the Itaewon neighborhood; at least 172 additional people were hospitalized. Officials determined that one survivor's suicide death in December 2022 was due to the tragedy, making him the 159th victim by the law in South Korea.
     In this study, we will be applying convolution neural network models to see the occupancy of the building from time to time and able to monitor the public flow into and out of the building at the same time, we use the latest version of the CNN model YOLO algorithm and the optimizers of our choice SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp based on the best performance to detect the Objects in the frame like persons, dogs, cats, handbags, chairs etc. The continuation of these sections will be Section 2 which addresses the related the related work of our study, Section 3 Methodology shows and explains our approaches and methods of working, Section 4 describes the results and Section Conclusion then we conclude the paper with proper references at the ending.

Objectives
The following are the objectives of this project:
Enable real-time monitoring of foot traffic and animal presence within buildings, providing instant and up-to-date information for decision-making.
Design the technology to be scalable, capable of handling varying building sizes and types, from small offices to large shopping malls or educational campuses.
Develop solutions that are non-intrusive to the occupants of the buildings, respecting privacy while still delivering accurate data.
Ensure seamless integration with existing building infrastructure, including surveillance systems, access control systems, and other relevant technologies.
Account for diverse environmental conditions such as varying lighting, weather, and building layouts to ensure the technology's effectiveness in different settings.
Develop a solution that is cost-effective both in terms of initial implementation and long-term operational expenses, providing value for the investment.




Background and Literature Survey

In the fields of AI, Deep learning, Computer vision, etc. Object detection is a basic study area where it has a majority of the complex tasks are done by it. It locates the targets based on the area of interest taken by the picture or frames in the video and replies with the bounding boxes[6]. The idea of object detection was first introduced in 2014 with the first pre-sized model, the R-CNN, which had an average presession percentage of objects at the time of capture (mAP) that was 53.7% according to PASCAL's VOC 2010 [7]. The models consist of several parts such as modal design, test-time detection, training, and evaluation, by then we will be able to get the map values based on the evaluated score. In the list of multiple detection modes R-CNN, Fast R-CNN, Faster R-CNN, SPP-net, R-FCN, FPN and Mask R-CNN, these are based on region proposal, where some of other model based on classification or regression like DSOD [11], MultiBox, AttentionNet, SSD, G-CNN, YOLOv2 [8], DSSD[10] and YOLO [9].
     Among them, in this paper, we are using the regression-based model named YOLO to detect the object and we explore the latest versions of it. This is a regression game in which we get bounding box and class probabilities straight from the image pixel. We use our models to predict the objects from those image pixels. The main reason for choosing this YOLO model is, that this is an extremely fast algorithm as this algorithm detects a regression problem we don’t need any complex pipelines when compared with others. YOLO sees the whole frame or picture at the time of training or testing so it interprets the relevant information about the classes and their appearance. When compared to Fast R-CNN which is a top detection model, YOLO creates less than half the amount of background mistakes. YOLO can recognize and recognize objects in a wide range of ways. It's way better than other detection algorithms, like DPM or R-CNN when it's trained on real photos and compared to artwork. Plus, since it's so versatile, it's less likely to mess up when you're trying to figure out what something looks like or when you enter something unusual[9].
     The YOLO framework is composed of five bounding boxes, each of which is expected to have a lattice unit at its focal point. The bounding boxes are expected to be affected by the quality of the entire image. The structure of YOLO is composed of three main components: the backbone, the neck, and the head. For the Aquatic dataset, the YOLOov5 model achieved a mAP of 0.84, the Sign Language dataset of 0.87, the Chessboard dataset of 0.9, and the Library Books dataset of 0.86. This demonstrates that version 5 of YOLO works well for the detection. For the Racoon dataset, this model achieved the highest mAP with an accuracy of 91%. In this paper, we are going to use the latest version of the YOLO which is a YOLOv8 that was launched in 2023 2 years after launching v5, with major improvements and can do detection, segmentation, and classification [13].


Fig. 1: Architecture of YOLO
 







1.3 	Organization of the Report

The remaining chapters of the project report are described as follows:
Chapter 2 contains the proposed system, methodology and software details.
Chapter 3 gives the cost involved in the implementation of the project.
Chapter 4 discusses the results obtained after the project was implemented.
Chapter 5 concludes the report.
Chapter 6 consists of codes.
Chapter 7 gives references.



CHAPTER 2
OccupEye -Building Traffic and Animal Monitoring System

This Chapter describes the proposed system, working methodology, software details. 

2.1 Proposed System
The proposed method employs video frames to count persons and detect animals in public locations, including banks, government offices, parks, eat street etc. we utilize the YOLO v8 pretrained model for person counting and animal detection in CCTV videos. Leveraging the Line of Interest (LOI) method, we track individuals' trajectories based on imaginary lines, classifying them as entering or exiting. The YOLO v8 model, with its real-time processing and object detection capabilities, aids in precise identification and tracking of persons and animals. The geographical angle of camera frames determines directional movement, essential for accurate counting. Integration of person counting and animal detection provides a comprehensive overview. Alerts are triggered for animals entering buildings, ensuring timely responses. Measured data is displayed on-screen and transmitted when specific conditions are met, enhancing security and resource management.


Fig. 2: System Block Diagram






ARCHITECTURE








Fig 3: Architecture












2.2 	Working Methodology

For our operations and performance work we are going to use the YOLOv8 which is the latest version in the YOLO family with cutting-edge performance in accuracy and speed. This latest model was trained on the COCO 2017 Dataset.MS COCO full name Microsoft Common Object in context is a dataset consisting of around 328K Images. When comparing COCO with ImageNet, PASCAL VOC 2012, SUN. This additional dataset consists of a variety of datasets, ranging from small to large, with a range of categories and types of images. The primary purpose of PASCAL Visualization and Analysis (PASCAL VOC) is to identify objects in natural images, while SUN focuses on recognizing different types of scenes and the features that are commonly associated with them. MS COCO's purpose is to identify and segment objects as they are encountered in their natural environments[14]. Even though Microsoft Cognos Object-Oriented Computing (COCO) has fewer categories than the SUN and ImageNet, it still has a higher number of predicted instances per category, which will be advantageous for more complex models. Generally, the dataset of COCO typically has a 7.7 instance count per image and a 3.5 instance count per category. On the other hand, ImageNet and the PASCAL Visualization Object Order (POVO) has an average of less than two and three instances respectively per image, as illustrated in Fig.2 [14].

Fig. 4: Number of Category vs Instances on Different Datasets

     In this section, we propose that our main work Is on the concept of LOI ,In this method, we are counting the objects that crossed the imaginary line. We track the trajectory of the person using two lines and classify the person as entering or exiting the building. This line of Interest method helps us count the persons, We will be counting the people who pass the imaginary lines fixed at a certain position that everybody must cross through as shown in Fig. 4. The lines are marked using the frame height coordinates and extended to the frame width. Our YOLO model detects the Person or any object in the frame and provides us with the center coordinates of it. We use them to give a unique ID to every object that we are considering. Now these ID plays a key role in our work as we are going to count these IDs as persons and calculate the no of people who entered or exited the building. The Id is determined as Entered or Exited if and only if it crosses both the Imaginary Lines marked by us [16].
     Here the geographical angle of the frame plays the key role in differentiating between the entering or exiting, For example, if the camera sensor is installer inside the building, the person walking towards the frame or the ID crossing the Lines toward the frame means they are entering and the Exiting will be the person going away in the frame. And the phenomenon is reversed if the camera sensor is installed outside of the building. Based on this phenomenon we count the people entering or Exiting the building.
     In our work other than Persons, animals are also detected and Identified with a unique ID, and now in the case of an animal, if the animal is inside the building or entered the building the Animal is marked with a Bounding box for the whole time it was inside. A System-generated Alert will be called to the corresponding security personnel when any animal enters to building and respective measures will be taken care of. The Measured Date calculated by our algo will be displayed on the screen itself as shown in Fig. 4 in the results section. The same data will be sent to corresponding Officers if the counted persons in the building reach the Limit of the Building capacity based on the architecture of the building. The results and discussion section will discuss all the results with samples.

2.3 Standards
 Various standards used in this project are:
General Data Protection Regulation (GDPR): In European countries, GDPR sets guidelines for the processing of personal data. Ensure that your system complies with GDPR principles, including obtaining consent, anonymizing data, and providing individuals with control over their data.
Surveillance Camera Codes of Practice:
Some countries may have specific guidelines or codes of practice for the use of surveillance cameras. For instance, the UK has a Surveillance Camera Code of Practice that provides guidance on the use of surveillance cameras by government and law enforcement.
Audio Recording Regulations:
If your CCTV system includes audio recording, be aware of the legal implications. In many jurisdictions, audio recording may have stricter regulations than video recording.


2.4 SDLC
	In our project, we use the waterfall model as our software development life cycle because of its step-by-step procedure while implementing.
Fig. 5: Waterfall Model
Gathering and analyzing requirements - All possible requirements for the system being developed are collected in this phase and documented in the requirements specification.
System design - the specifications required from the first phase are checked in this step and the system design is prepared. This system design specifies hardware and system requirements and helps define the overall system architecture.
Implementation - Using system design inputs, the system is first developed in small programs called units, which are integrated in the next phase. Each unit is developed and tested for performance. This is called unit testing.
Integration and testing - all developed units are integrated into the system during the implementation phase after each unit is tested. After integration, the entire system is tested for defects and failures.
System Deployment – Here completion of functional and non-functional testing take place. 
Maintenance - Some issues occur in the user's environment. Patches will be released to fix these issues. Maintenance is done to respond to these changes in the user environment.

FEASIBILITY STUDY:
At this stage, the feasibility of the project is analyzed and a business proposal is presented along with a very general outline of the project and some cost estimates. During system analysis, a feasibility study of the proposed system is conducted. This is to ensure that the proposed system is not a burden on businesses. For feasibility analysis, it is essential to have a good understanding of the key system requirements.
Three important points related to feasibility analysis are:
Economic feasibility
Technical feasibility
Social feasibility
Economic feasibility:
This study was conducted to determine the economic impact of the system on the organization. There is a limit to the amount of capital that companies can invest in system research and development. Spending must be justified. Therefore, we were able to develop this system within the budget. This was achieved because most of the technology used is freely available. Only custom products had to be purchased.
Technical feasibility:
This study was conducted to check the technical feasibility of the system, that is, the technical requirements. The developed system should not make too much demand on the available technical resources. This places great demands on available technical resources. This creates many demands for customers. The requirements of the developed system should be moderate, since the implementation of this system requires only minimal or zero changes.
Social feasibility:
This aspect is to determine the level of acceptance of the system by users. It involves the process of training users to use the system effectively. Users should not feel threatened by the system and should accept it if necessary. The level of user acceptance depends solely on the methods adopted to educate and familiarize users with the system. Since he is the end user of the system, his confidence level should be increased so that he can also give constructive criticism, which is welcome.

2.5   System Requirements Specification 
Functional and non-functional requirements:
System requirements in a project refer to the specifications and criteria that define the capabilities, functionalities, and performance characteristics expected from the system being developed. These requirements serve as a foundation for the design, development, testing, and implementation of the project. System requirements typically cover various aspects to ensure the successful delivery of a functional and reliable system. Here are key components of system requirements:
Functional Requirements:
Functional requirements articulate the specific features and capabilities that the system must possess to fulfill the project's objectives. These requirements essentially describe what actions the system should perform and the outcomes it should deliver. They form the backbone of the system's functionality and directly address the "what" aspect of the system's behavior. For instance, in an e-commerce system, a functional requirement might specify the ability to add products to a shopping cart, process payments, and generate order confirmations.
Non-Functional Requirements:
Non-functional requirements complement functional requirements by focusing on aspects beyond specific functionalities. These criteria encompass performance, reliability, usability, and security considerations. Unlike functional requirements that address what the system should do, non-functional requirements emphasize how well the system should perform its functions. For example, a non-functional requirement might dictate that the system should respond to user inputs within two seconds (performance), maintain data integrity (reliability), provide an intuitive user interface (usability), and adhere to specified security standards.
The functional requirements outline the specific actions and features a system should have, while non-functional requirements specify how well the system should perform these actions, considering broader aspects like performance, reliability, usability, and security. Both types of requirements are integral to developing a comprehensive understanding of the system's scope and ensuring its successful design, development, and implementation.
SYSTEM SPECIFICATIONS:
H/W Specifications:
Processor	: I3/Intel Processor
RAM	: 8GB (min)
Hard Disk	: 128 GB
S/W Specifications:
Operating System	: Windows 10
Server-side Script	: Python 3.6
IDE	: Spyder
Libraries Used	: Numpy, cv2, ultralytics, Pandas

2.6 MODULES:
	UML DIAGRAMS:

UML stands for Unified Modelling Language. UML is a standardized general-purpose modelling language in the field of object-oriented software engineering. The standard is managed, and was created by, the Object Management Group.
In its current form UML is comprised of two major components: a Meta-model and a notation. In the future, some form of method or process may also be added to; or associated with, UML.
The Unified Modelling Language is a standard language for specifying, Visualization, Constructing and documenting the artifacts of software system, as well as for business modelling and other non-software systems.
The UML represents a collection of best engineering practices that have proven successful in the modelling of large and complex systems.
The UML is a very important part of developing objects-oriented software and the software development process. The UML uses mostly graphical notations to express the design of software projects.

GOALS:
The Primary goals in the design of the UML are as follows:
Provide users a ready-to-use, expressive visual modelling Language so that they can develop and exchange meaningful models.
Provide extendibility and specialization mechanisms to extend the core concepts.
Be independent of particular programming languages and development process.
Provide a formal basis for understanding the modelling language.
Encourage the growth of OO tools market.
Support higher level development concepts such as collaborations, frameworks, patterns and components.
Integrate best practices.


USE CASE DIAGRAM:

Fig. 6: Use Case Diagram



ACTIVITY DIAGRAM:


Fig. 7: Activity Diagram


SEQUENCE DIAGRAM:





Fig. 8: Sequence Diagram




 ER DIAGRAM:



Fig. 9: ER Diagram


 CLASS DIAGRAM:



Fig. 10: Class Diagram


2.6	System Details

This section describes the software and details of the system:

2.6.1 	 Software Details 


+SOFTWARE INSTALLATION FOR MACHINE LEARNING PROJECTS:

Installing Anaconda:

To install Anaconda, the user can follow these steps:

Download Anaconda:
   - The user should visit the official Anaconda website and download the Windows installer
   - Locate and double-click on the downloaded installer file.

Fig. 11: Different Versions of Anaconda available in internet

   - Follow the on-screen instructions, clicking "Next" through the installation screens.
   - Choose the default installation location and click "Next."
   - Click "Install" to initiate the installation process.
   - Check the option "Add Anaconda to my PATH" and click "Next."
   - Open Anaconda Navigator:
   - Launch Anaconda Navigator from the Start menu.
- Open Anaconda Prompt and type `conda list` to verify installed packages.

For Spider installation:
   - If Anaconda is not installed, follow the steps mentioned above for Anaconda installation on Windows 10.
   - Once Anaconda is installed, open Anaconda Navigator from the Start menu.

Fig 12: Anaconda Navigator
   - In Anaconda Navigator, navigate to the "Home" tab.
   - Find the "IDEs" section and click "Launch" next to Spider.
   - Open the Anaconda Prompt from the Start menu.
   - Type `conda install spider -c conda-forge` and press Enter.
   - Conda will download and install Spider along with its dependencies. Wait for the process to complete.
   - Once the installation is complete, the user can either launch Spider from the Anaconda Navigator or by typing `spider` in the Anaconda Prompt.
   - After launching Spider, the user should see the Spider IDE window. Testing can be done by writing a simple Python script or using the interactive console.

Fig. 13: Installing Spyder using cmd

  You need to install some packages to execute your project in a proper way.
Open the command prompt/ anaconda prompt or terminal as administrator.
The prompt will get open, with specified path, type “pip install package name” which you want to install (like NumPy, pandas, sea born, scikit-learn, Matplotlib, Pyplot)

Ex: Pip install NumPy

- For Conda Users:
  - If using conda, the user can execute the command: `conda install numpy`.

Fig. 14.1: Installing numpy library

- For PIP Users:
  - Users preferring pip can use the command: `pip install numpy`.

Fig. 14.2: Installing Numpy Library

These steps ensure a smooth installation of Anaconda, Spider IDE, and NumPy, providing users with a powerful environment for Python development and scientific computing.














CHAPTER 3
COST ANALYSIS

3.1	List of components and their cost
The costs of the various components used in this project are given below in Table 3.1.

Table 3.1 List of components and their costs

CHAPTER 4

RESULTS AND DISCUSSIONS

The pre-trained model of the YOLO we had used of the latest version 8 named YOLOv8 comes with different variations like nano, small, medium, large, and extra large which was built by the dark web community. This model with the coco dataset got the higher mAP among the other versions with comparing to parameters and Latency A100 TensorRT FP16(ms/img) as shown in Fig. 3 [15].
  
Fig. 15: Representation of mAP Val of different versions of YOLO on COCO

We have used the extra large model for our work as it stands with a high mAP Val among all the variations available for us. As of the Fig.3 we can see the mAP of the Yolov8x model is around 54mAP and the sample result of the model is also shown below. With all the efforts we finally got the person detected, assigned with unique IDs, and counted when crossed the imaginary lines. The line of intersection phenomenon helps us to determine whether the personnel entered or exited from the building. All the sample images are represented below. 




Fig. 16: Perfomance Output of YOLOv8x on an image


Fig. 17: Entrance of the building and Bounding box around the person while he entered the building.
In this image, we can see that a person is entering a building where there are a lot of people around, Only when the person crosses both lines, then the count be considered, and the Persons inside the building increase by 1. The same process will be continued till the code is stopped manually when it is implemented with the live stream. Here are some more samples.
The Total persons in the Buildings are calculated with the formulae:

Total Persons in The building = (Persons In – Persons Out) 		   (1)



Fig. 18: People entering the building



Fig. 19: People exiting the building.


People are exiting the building rapidly within seconds, but our model is capable enough to keep an eye on every person on the way. From Fig. 5, 10 personals exited whereas only 1 person entered, the final persons inside the building were in negative which was calculated with the formulae(1) , This type of scenario will be caused when the building has multiple entrances and the video source is from the middle of the rush hour. When the animals entered the building the bounding box was shown in red for the whole time until the animal was sent out. The same is shown in the figure below.



Fig. 20:  Animals entering a store to have some food and detected along with a visual representation of a Bounding Box around the animal as long as it stays inside.
In this sample, while the dogs entered the store for food they were marked with a Red Bounding box, and a visual alert was also shown named “Dog IN ALERT”. And a sound alert will also be triggered to the shopkeeper to alert him about the dogs.




CHAPTER 5

CONCLUSION AND FUTURE WORK


By the Year 2023, the world will be advancing with Artificial Intelligence, where every machine helps humans a lot. We managed to perform the detection and surveillance operation using the latest version of the YOLO i.e., YOLOv8. The impact of YOLOv8 extends far beyond crowd management, it has ushered in a new era of computer vision applications. These applications have found their place in a myriad of public spaces, such as stadiums, airports, and event venues, ensuring that the safety and comfort of attendees are prioritized. This remarkable technology has allowed us to efficiently monitor and record accurate crowd data. The versatility of YOLOv8 is further evident in its ability to detect intrusions. This capability has proven invaluable in alerting the presence of animals or unauthorized individuals within buildings and restricted areas, fortifying security measures and deterring unwanted disruptions.  By the work that we have done, people can be saved by restricting them to gather at the same place with huge strength. We managed to bring the computer vision application with the detection algorithm and successes by monitoring the accurate values of the strength of a crowd. This type of application can be very useful in all public places, where a lot of different people join together. Recently we have had so many incidents where animals attacked local villagers. Through this project, We successfully managed to alert the presence of animals inside the building or place, which will be useful in avoiding unwanted disruption and human safety. 
     In the future, we are going to take this work to the next level by connecting multiple video sources & processing simultaneously, 3𝐷 localization, and a depth camera will be developed. Also, more advanced DL object detectors will be trained in our custom dataset and compared with YOLOv8. All entrances are tracked and can be processed with a single system and we plan to connect the feed directly to the cloud infrastructure, where the data will be directly reflected on the web portal we are planning to design and all the statistics of the entry flow will be recorded and presented in the form of Graphs and even available to filter for research purpose. This Data can be shared with the relevant government departments such as Police, firefighters, etc., where they can also perform manual security monitoring of the building. It's important to underline that while AI, particularly YOLOv8, plays a pivotal role in security and surveillance, it is not a replacement for human oversight. Instead, it complements manual security monitoring, offering real-time insights and alerts to security personnel, resulting in a more robust security infrastructure.
     In conclusion, YOLOv8 and AI technologies have not just evolved but revolutionized our surveillance and security systems, significantly advancing our ability to manage crowds, detect intrusions, and maintain public safety. The future holds the promise of even more seamless integration, enhanced data accessibility, and deeper collaboration with government agencies, underscoring the transformative power of AI in enriching lives and ensuring the safety of communities worldwide.
