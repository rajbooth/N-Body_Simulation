// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp> 

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

// TIFF library
#include <tiff.h>
#include <tiffconf.h>
#include <tiffio.h>
#include <tiffvers.h>

#include "Element.h"
#include "SineWave.h"
#include "Mesh.h"
#include "Particles.h"
#include "SimParams.h"

#define REFRESH_DELAY     10 //ms

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 960;
const unsigned int window_height = 640;

// display parameters
float meshIntensity = 0.0f;
float meshAlpha = 1.0;
float particleAlpha = 0.5;
int pixelSize = 1;
bool slice = false;
float sliceWidth = 0;
float sliceZ = 0;
bool pause = false;
bool saveOn = false;
bool meshRandom = false;
bool displayHisto = false;
bool captureHisto = false;
bool centreHisto = false;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int frame = 0;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

// Elements
SineWave* sine;
Mesh* mesh;
Particles* parts;

// Simulation parameters
SimParams m_params;

//Simulation statistics
PhiStats m_pstats;
ParticleStats m_vstats;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
bool saveSnapshot(int, int);
void drawBitmapText(char *string, int x, int y);
void displayVelocityGraph();
void setParams(int mode);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void functions(int key, int x, int y);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void wheel(int, int, int, int);
void timerEvent(int value);

const char *sSDKsample = "Particle-Mesh N-Body Simulation";

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	printf("%s starting...\n", sSDKsample);

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}

		if (checkCmdLineFlag(argc, (const char **)argv, "sphere"))
		{
			setParams(1);
		}
		else if (checkCmdLineFlag(argc, (const char **)argv, "mega"))
		{
			setParams(2);
		}
		else setParams(0);
	}
	else
	{
		setParams(0);
	}

	printf("\n");

	runTest(argc, argv, ref_file);

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void setParams(int mode)
{
	m_params.meshSize = 64;
	m_params.meshMode = 1; // 0 = lattice, 1 = random
	m_params.meshCellSize = 2.0 / m_params.meshSize;
	m_params.numParticles = 100000;
	m_params.particleMass = 1.0;
	m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 8;	//use 8 for normal run, 1 for galaxy collapse
	m_params.initialSphere = 4;												//use 4 for normal run, 0.5 for galaxy collapse
	m_params.overDensity = 0.0;
	m_params.velocityScaling = 0.001;
	m_params.G = 0.5;							// Changing from 1.0 to 0.1 slower evolution.  More string structure
	m_params.integrateTimestep = 0.001f;
	m_params.diffusionTimestep = 0.5f;			// decreasing this from  1.0 to 0.1 reduces gravitational clumping and leads to many more smaller galaxies
												// needs to be <= 1.0 otherwise phi blows up

	switch (mode)
	{
		case 1:  //collapse overdensity sphere
			m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 1;	//use 1 for galaxy collapse
			m_params.initialSphere = 0.5;
			m_params.overDensity = 0.2;
			m_params.G = 1.0;
			m_params.diffusionTimestep = 0.8;
			break;
		
		case 2:  //mega
			m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 8;	//use 8 for normal run, 1 for galaxy collapse
			m_params.numParticles = 1000000;
			m_params.particleMass =  0.1;
			m_params.G = 1.0;
			m_params.diffusionTimestep = 0.01f;
			m_params.integrateTimestep = 0.0001f;
			break;

		case 3: //high res grid
			m_params.meshSize = 128;
			m_params.initialSphere = 8;
			m_params.integrateTimestep = 0.0005f;
			m_params.diffusionTimestep = 0.1f;
			break;

		case 4: //high res grid and sphere
			m_params.meshSize = 128;
			m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 1;	//use 1 for galaxy collapse
			m_params.initialSphere = 1.0;
			m_params.overDensity = 0.1;
			m_params.numParticles = 500000;
			m_params.G = 1.0;
			m_params.integrateTimestep = 0.001f;
			m_params.diffusionTimestep = 0.5f;
			break;

		case 5:  //mega + high res grid
			m_params.meshSize = 128;
			m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 8;	//use 8 for normal run, 1 for galaxy collapse
			m_params.numParticles = 1000000;
			m_params.particleMass = 0.1;
			m_params.G = 1.0;
			m_params.diffusionTimestep = 0.05f;
			m_params.integrateTimestep = 0.0005f;
			break;

	default:
			 m_params.potentialLimit = m_params.meshSize * m_params.meshSize * 8;	//use 8 for normal run, 1 for galaxy collapse
			 m_params.initialSphere = 4;												//use 4 for normal run, 0.5 for galaxy collapse
			 m_params.overDensity = 0.0;
	}
}

void restart(int mode)
{
	// reset initial conditions
	pause = true;
	setParams(mode);
	delete mesh;
	delete parts;
	mesh = new Mesh(&m_params);
	parts = new Particles(&m_params, mesh);
	pause = false;
}

void computeFPS()
{
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	if (saveOn)
	{
		frameCount++;
		sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz), saving frame %01d", avgFPS, frameCount);
	}
	else
	{
		sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	}
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutMouseWheelFunc(wheel);
	glutCloseFunc(cleanup);
	glutSpecialFunc(functions);

	// initialize necessary OpenGL extensions
	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	SDK_CHECK_ERROR_GL();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
		{
			return false;
		}
	}
	else
	{
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMouseWheelFunc(wheel);
	glutCloseFunc(cleanup);
	glutSpecialFunc(functions);

	//sine = new SineWave();	
	mesh = new Mesh(&m_params);
	parts = new Particles(&m_params, mesh);

	// start rendering mainloop
	glutMainLoop();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);
	// Increment frame counter
	frame++;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Projection matrix : 30° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(glm::radians(30.0f), (float)window_width / (float)window_height, 0.1f, 100.0f);

	// Camera matrix
	glm::mat4 View = glm::lookAt(
		glm::vec3(0, 0, 2), // Camera is at (0,0,3), in World Space
		glm::vec3(0, 0, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
	);

	// Model matrix : an identity matrix (model will be at the origin)
	//glm::mat4 Model = glm::mat4(1.0f);
	glm::mat4 Model = glm::translate(glm::vec3(0.0, 0.0, translate_z));
	Model = glm::rotate(Model, rotate_x, glm::vec3(1, 0, 0));
	Model = glm::rotate(Model, rotate_y, glm::vec3(0, 1, 0));

	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 mvp = Projection * View * Model; // Remember, matrix multiplication is the other way around

	// Run CUDA kernals
	if (!pause)
		parts->execCUDA();

	//sine->Draw();
	mesh->Draw(mvp);
	parts->Draw(mvp);

	if (frame % 20 == 0)  //Update mesh statistics every 1 seconds
	{
		m_pstats = mesh->stats;
		m_vstats = parts->vstats;
	}
	// Display mesh phi statistics
	char msg[20];
	float delta = (m_pstats.max_phi - m_pstats.avg_phi) / m_pstats.avg_phi;
	if (isnan(delta)) delta = 0;
	sprintf(msg, "Delta phi = %0.3f",delta);
	drawBitmapText(msg, window_width - 120, 10);
	sprintf(msg, "Avg phi = %0.1f", m_pstats.avg_phi);
	drawBitmapText(msg, window_width - 120, 30);
	sprintf(msg, "Max phi = %0.1f", m_pstats.max_phi);
	drawBitmapText(msg, window_width - 120, 50);
	sprintf(msg, "Min phi = %0.1f", m_pstats.min_phi);
	drawBitmapText(msg, window_width - 120, 70);

	// Display particle velocity statistics
	sprintf(msg, "Avg vel = %0.4f", m_vstats.avg_v);
	drawBitmapText(msg, 10, 10);
	sprintf(msg, "Max vel = %0.2f", m_vstats.max_v);
	drawBitmapText(msg, 10, 30);
	sprintf(msg, "Min vel = %0.4f", m_vstats.min_v);
	//drawBitmapText(msg, 10, 50);

	// Display velocity graph
	if (displayHisto)
	{
		if (frame % 50 == 0)
			parts->calcStatistics(frame);
		displayVelocityGraph();
	}

	// Save window to TIFF file
	if (saveOn)
		saveSnapshot(window_width, window_height);

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	delete sine;
	delete mesh;
	delete parts;

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard special functions handler
////////////////////////////////////////////////////////////////////////////////
void functions(int key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (GLUT_KEY_F1): //F1 function key.
		restart(1);
		break;

	case (GLUT_KEY_F2): //F2 function key.
		restart(2);
		break;

	case (GLUT_KEY_F3): //F3 function key.
		restart(3);
		break;

	case (GLUT_KEY_F4): //F4 function key.
		restart(4);
		break;

	case (GLUT_KEY_F5): //F5 function key.
		restart(5);
		break;

	case (GLUT_KEY_F6): //F6 function key.
		break;
	case (GLUT_KEY_F7): //F7 function key.
		break;
	case (GLUT_KEY_F8): //F8 function key.
		break;
	case (GLUT_KEY_F9): //F9 function key.
		break;
	case (GLUT_KEY_F10): //F10 function key.
		restart(10);
		break;

	case (GLUT_KEY_F11): //F11 function key.
		break;
	case (GLUT_KEY_F12): //F12 function key.
		break;

	case (GLUT_KEY_UP): // up-arrow
		sliceZ += 0.1f;
		break;

	case (GLUT_KEY_DOWN): // down-arrow
		sliceZ -= 0.1f;
		break;

	case(GLUT_KEY_LEFT):
	case(GLUT_KEY_RIGHT):
	case(GLUT_KEY_PAGE_UP):
	case(GLUT_KEY_PAGE_DOWN):
	case(GLUT_KEY_HOME):
	case(GLUT_KEY_END):
	default:
	std: cout << "Keypress " << key << "\n";
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case 'a':
		if (meshAlpha == 0.0f)		meshAlpha = 1.0f;
		else meshAlpha = 0.0f;
		break;

	case 'c': //centre histogram
		centreHisto = !centreHisto;
		break;

	case 'm': // make mesh visible
		if (meshIntensity == 0.0f)		meshIntensity = 0.5f;
		else meshIntensity = 0.0f;
		break;

	case 'n': //add noise to mesh
		meshRandom = !meshRandom;
		mesh->SetVertices(m_params.meshCellSize, meshRandom);
		return;

	case 'h':
		displayHisto = !displayHisto;
		if (displayHisto) captureHisto = true;
		break;

	case 'v':
		if (particleAlpha == 0.0f)		particleAlpha = 1.0f;
		else particleAlpha = 0.0f;
		break;

	case 's':
		if (sliceWidth == 0.0f)
			sliceWidth = 0.1f;
		else sliceWidth = 0.0f;
		break;

	case 'r': //record TIFF file
		saveOn = !saveOn;
		break;

	case 'p':
		pause = !pause;
		break;

	case '+':
	case '=':
		sliceWidth += 0.1f;
		break;

	case '-':
		sliceWidth -= 0.1f;
		break;

	case '0':
		pixelSize = 0.0f;
		break;

	case '1':
		pixelSize = 1.0f;
		break;

	case '2':

		pixelSize = 2.0f;
		break;

	case '3':
		pixelSize = 3.0f;
		break;

	case (27):
	case 'q':
		glutDestroyWindow(glutGetWindow());
		return;

	default:
	std: cout << "Keypress " << key << "\n";
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	if ((button == 2) || (button == 3) || (button == 4)) // It's a wheel event
	{
		// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
		if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
		printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
	}
	else {  // normal button event
		printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.02f;
		rotate_y += dx * 0.02f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void wheel(int button, int dir, int x, int y)
{
	if (dir > 0)
	{
		// Zoom in
	}
	else
	{
		// Zoom out
	}

	return;
}

bool saveSnapshot(int width, int height)
{

	bool ret = false;
	TIFF *file;
	GLubyte *image, *p;
	int i;
	char out_path[200], out_fname[200];

	sprintf(out_path, "D:/Cosmological Data/PMsim/");
	sprintf(out_fname, "%sSnapshot%05d.tif", out_path, frame);

	file = TIFFOpen(out_fname, "w");
	if (file) {
		image = (GLubyte *)malloc(width * height * sizeof(GLubyte) * 3);

		/* OpenGL's default 4 byte pack alignment would leave extra bytes at the
		end of each image row so that each full row contained a number of bytes
		divisible by 4.  Ie, an RGB row with 3 pixels and 8-bit componets would
		be laid out like "RGBRGBRGBxxx" where the last three "xxx" bytes exist
		just to pad the row out to 12 bytes (12 is divisible by 4). To make sure
		the rows are packed as tight as possible (no row padding), set the pack
		alignment to 1. */
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, image);
		TIFFSetField(file, TIFFTAG_IMAGEWIDTH, (uint32)width);
		TIFFSetField(file, TIFFTAG_IMAGELENGTH, (uint32)height);
		TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 8);
		TIFFSetField(file, TIFFTAG_COMPRESSION, COMPRESSION_PACKBITS);
		TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
		TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 3);
		TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(file, TIFFTAG_ROWSPERSTRIP, 1);
		TIFFSetField(file, TIFFTAG_IMAGEDESCRIPTION, "");
		p = image;
		for (i = height - 1; i >= 0; i--) {
			if (TIFFWriteScanline(file, p, i, 0) < 0) {
				free(image);
				TIFFClose(file);
				return false;
			}
			p += width * sizeof(GLubyte) * 3;
		}
		TIFFClose(file);
	}
	return ret;
}

void drawBitmapText(char *string, int x, int y)
{
	char *c;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glm::mat4 orth = glm::ortho(0.0f, (float)window_width, 0.0f, (float)window_height);
	glMultMatrixf(&(orth[0][0]));
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_BLEND);
	glUseProgram(0);
	glColor3f(0.0, 1.0, 1.0);
	glRasterPos2i(x, y);

	for (c = string; *c != NULL; c++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void displayVelocityGraph()
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glm::mat4 orth = glm::ortho(0.0f, (float)window_width, 0.0f, (float)window_height);
	glMultMatrixf(&(orth[0][0]));
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_BLEND);
	glUseProgram(0);


	int x, y, x0 = 20, y0 = 550;
	if (centreHisto)
	{
		x0 = window_width / 2;
		y0 = window_height / 2;
	}

	glLineWidth(3);
	glBegin(GL_LINE_STRIP);
	glColor3f(0.0, 0.0, 0.0);
	for (int i = 0; i < 24; i++)
	{
		x = i * 8 + x0;
		y = y0 + parts->vstats.velocity[i] * 2000;
		glVertex2f(x,y);
	}
	glEnd();

	glLineWidth(1);
	glBegin(GL_LINE_STRIP);
	glColor3f(0.0, 1.0, 1.0);
	for (int i = 0; i < 24; i++)
	{
		x = i * 8 + x0;
		y = y0 + parts->vstats.velocity[i] * 2000;
		glVertex2f(x, y);
	}
	glEnd();

	glLineWidth(3);
	glBegin(GL_LINES);
	glColor3f(0.0, 0.0, 0.0);
	glVertex2f(x0, y0);
	glVertex2f(x0 + 240, y0);
	glVertex2f(x0, y0);
	glVertex2f(x0, y0 + 90);
	glEnd();

	glLineWidth(1);
	glBegin(GL_LINES);
	glColor3f(1.0, 1.0, 1.0);
	glVertex2f(x0,y0);
	glVertex2f(x0 + 240, y0);
	glVertex2f(x0, y0);
	glVertex2f(x0, y0 + 90);
	glEnd();

	for (int i = 0; i < 32; i += 2)
	{
		glBegin(GL_LINES);
		glColor3f(1.0, 1.0, 1.0);
		glVertex2f(i * 8 + x0, y0 + 2);
		glVertex2f(i * 8 + x0, y0 - 2);
		glEnd();
	}

	glRasterPos2i(x0-10,y0+50);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'v');
	glRasterPos2i(x0+100, y0-20);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'r');

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}