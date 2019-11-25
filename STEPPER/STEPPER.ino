#include <Stepper.h>
const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution
Stepper myStepper(stepsPerRevolution, 2, 3, 4, 5);

String lenS = "SS";
int len = 0;

int globSteps = 0;
int safety = 0;

void setup() {

  pinMode(13, OUTPUT);
  myStepper.setSpeed(100);
  Serial.begin(9600);
  pinMode(A0, INPUT);

}

void loop() {

  if (Serial.available() > 0) {
    safety = 0;
    lenS = Serial.readStringUntil('\n');

    if (lenS == "rel") {
      myStepper.setSpeed(100);
      myStepper.step(globSteps);
      globSteps = 0;
      delay(500);
    } else {

      float a = -0.2848;
      float b = 0.6591;
      float c = 76.673;
      float B, x;
      int steps;

      len = lenS.toInt();
      c = c - float(len);
      B = pow(b, 2) - (4 * a * c);
      x = (-b - sqrt(B)) / (2 * a);

      steps = round(x / 0.04);

      for (int i = 0; i <= steps; i++) {
        if (analogRead(A0) < 900) {
          safety = 1;
          break;
        } else if (globSteps >= 400) {
          delay(1000);
          myStepper.step(globSteps);
          delay(1000);
          globSteps = 0;
          safety = 1;
          break;
        }
        myStepper.step(-1);
        globSteps++;
        delay(10);
      }

      while (analogRead(A0) > 900 && safety == 0 && globSteps < 400) {
        myStepper.setSpeed(20);
        myStepper.step(-10);
        globSteps = globSteps + 10;
        delay(10);
      }
      if (globSteps >= 400) {
        myStepper.step(globSteps);
        delay(1000);
        globSteps=0;
      }
    }
  }
  digitalWrite(13, LOW);
  //Serial.println(analogRead(A0));

}
