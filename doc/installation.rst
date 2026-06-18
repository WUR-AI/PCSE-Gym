Installation instructions
-------------------------

CropGym requires Python 3.10–3.12 and `PCSE <https://github.com/ajwdewit/pcse.git>`__ 6.x.

Using Poetry (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone `CropGym <https://github.com/WUR-AI/PCSE-Gym>`__
2. Install with Stable-Baselines3 extras:

   .. code-block:: bash

      poetry install --extras sb-integration

Manual installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install pcse>=6.0.13 gymnasium pyyaml numpy
   pip install stable-baselines3 sb3-contrib matplotlib torch tensorboard scipy tqdm pandas
   pip install -e .

The code has been tested using Python 3.11.
