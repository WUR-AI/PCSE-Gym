.. CropGym documentation master file, created by
   sphinx-quickstart on Thu Mar  9 21:25:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CropGym: a Reinforcement Learning Environment for Crop Management
-----------------------------------------------------------------

CropGym is a highly configurable `Python Gymnasium <https://gymnasium.farama.org/>`__ environment to conduct Reinforcement
Learning (RL) research for crop management. CropGym is built around
`PCSE <https://pcse.readthedocs.io/en/stable/>`__, a well established
python library that includes implementations of a variety of crop
simulation models. CropGym follows standard gym conventions and enables
daily interactions between an RL agent and a crop model.

Installation
------------
.. toctree::
   :maxdepth: 2

   installation.rst

Examples
--------
.. toctree::
   :maxdepth: 2

   examples.rst

Use Cases
---------
.. toctree::
   :maxdepth: 2

   usecases.rst

Citing CropGym
--------------

If you use CropGym in your publications, please cite us following this Bibtex entry

.. code-block:: text

    @article{cropgym,
      title={Nitrogen management with reinforcement learning and crop growth models},
      volume={2},
      DOI={10.1017/eds.2023.28},
      journal={Environmental Data Science},
      publisher={Cambridge University Press},
      author={Kallenberg, Michiel G.J. and Overweg, Hiske and van Bree, Ron and Athanasiadis, Ioannis N.},
      year={2023},
      pages={e34}
    }

Contact
-------
:email:`info@cropgym.ai`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
