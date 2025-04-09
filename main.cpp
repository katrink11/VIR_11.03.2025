#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string detectShape(const vector<Point> &contour)
{
	double perimeter = arcLength(contour, true);
	double epsilon = max(3.0, 0.02 * perimeter);
	vector<Point> approx;
	approxPolyDP(contour, approx, epsilon, true);

	if (perimeter < 40)
		return "Noise";

	switch (approx.size())
	{
	case 3:
		return "Triangle";
	case 4:
	{
		Rect rect = boundingRect(approx);
		double aspectRatio = static_cast<double>(rect.width) / rect.height;
		double areaRatio = contourArea(approx) / (rect.width * rect.height);
		if (aspectRatio >= 0.85 && aspectRatio <= 1.15 && areaRatio > 0.85)
		{
			return "Square";
		}
		return "Rectangle";
	}
	case 5:
		return "Pentagon";
	case 6:
		return "Hexagon";
	default:
		double area = contourArea(contour);
		double circularity = 4 * CV_PI * area / (perimeter * perimeter);
		return (circularity > 0.8) ? "Circle" : "Oval";
	}
}

int main()
{
	system("chcp 65001 > nul");

	string video_path = "./video.mp4";
	VideoCapture cap(video_path);
	if (!cap.isOpened())
	{
		cout << "Не удалось открыть видеофайл!" << endl;
		return -1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);
	int delay = static_cast<int>(1000 / fps);

	VideoWriter writer("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

	Mat frame;
	while (cap.read(frame))
	{
		Mat gray, blurred, edges;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, blurred, Size(5, 5), 1.5);
		double otsu_thresh = threshold(blurred, edges, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
		Canny(blurred, edges, otsu_thresh * 0.5, otsu_thresh);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		Mat contourOutput = frame.clone();
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][2] == -1)
				continue;
			double minArea = frame.rows * frame.cols * 0.0005;
			if (contourArea(contours[i]) < minArea)
				continue;

			string shape = detectShape(contours[i]);
			if (shape == "Noise")
				continue;

			drawContours(contourOutput, contours, (int)i, Scalar(0, 255, 0), 2);

			Moments M = moments(contours[i]);
			if (M.m00 != 0)
			{
				int cx = static_cast<int>(M.m10 / M.m00);
				int cy = static_cast<int>(M.m01 / M.m00);
				putText(contourOutput, shape, Point(cx - 25, cy), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 100, 0), 2);
				circle(contourOutput, Point(cx, cy), 3, Scalar(0, 0, 255), -1);
			}
		}

		writer.write(contourOutput);
		imshow("Video Tracking", contourOutput);
		int key = waitKey(delay);
		if (key == 27)
			break;
		if (key == ' ')
			waitKey(0);
	}

	cap.release();
	writer.release();
	destroyAllWindows();
	return 0;
}