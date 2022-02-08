// helloWrapper.c
#include "Python.h"

extern int add(int, int);
extern void out(const char*, const char*);

/*
int add(int x, int y)
{
	return x + y;
}

void out(const char* adrs, const char* name)
{
	printf("こんにちは、私は %s の %s です。\n", adrs, name);
}
*/

static PyObject* hello_add(PyObject* self, PyObject* args)
{
	int x, y, g;

	if (!PyArg_ParseTuple(args, "ii", &x, &y))
		return NULL;
	g = add(x, y);
	return Py_BuildValue("i", g);
}

static PyObject* hello_out(PyObject* self, PyObject* args, PyObject* kw)
{
	const char* adrs = NULL;
	const char* name = NULL;
	static char* argnames[] = {"adrs", "name", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kw, "|ss",
			argnames, &adrs, &name))
		return NULL;
	out(adrs, name);
	return Py_BuildValue("");
}


static PyMethodDef hellomethods[] = {
	{"add", hello_add, METH_VARARGS},
	{"out", hello_out, METH_VARARGS | METH_KEYWORDS},
	{NULL},
};

// myModule definition struct
static struct PyModuleDef hello = {
    PyModuleDef_HEAD_INIT,
    "hello",
    "Python3 C API Module(Sample 1)",
    -1,
    hellomethods
};


/*
void inithello()
{
	Py_InitModule("hello", hellomethods);
}
*/

PyMODINIT_FUNC PyInit_hello(void)
{
    return PyModule_Create(&hello);
}