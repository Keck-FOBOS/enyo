Installation
============

Clone the repo
--------------

To download the software and associated data, clone the `GitHub repo
<https://github.com/Keck-FOBOS/enyo>`_ by executing:

    .. code-block:: bash

        git clone https://github.com/Keck-FOBOS/enyo..git

This will create an ``enyo`` directory in the current directory.

Install Python 3
----------------

``enyo`` is supported for Python 3 only. To install Python, you can
do so along with a full package manager, like `Anaconda
<https://www.continuum.io/DOWNLOADS>`_, or you can install python 3
directly from `python.org <https://www.python.org/>`_.


Install the code
----------------

To install ``enyo``, do one or more of the following (always from
within the top-level directory of the repo):

 * To perform an environment-level installation, run:

    .. code-block:: bash

        python3 setup.py install

 * On MacOSX, you may need to run:

    .. code-block:: bash

        CC=clang python3 setup.py install

 * To install ``enyo`` such that changes you make to the repo are
   immediately available in your environment, run:

    .. code-block:: bash

        python3 setup.py develop

 * To install ``enyo`` and ensure its dependencies are met, you can run:

    .. code-block:: bash

        pip3 install -e .

 * To install only the dependencies, run:

    .. code-block:: bash

        pip3 install -r requirements.txt

Problems?
---------

We have limited support to offer installation help. However, if you
have problems, particularly those that you think may be a more
general problem, please `submit an issue
<https://github.com/Keck-FOBOS/enyo/issues>`_.

