FROM mambaorg/micromamba:1.4.1

USER root
RUN apt update && apt install git gcc g++ -y # necessary for pip installs
USER mambauser

RUN micromamba install -n base --yes -c conda-forge python=3.10 pip ndcctools plumed cp2k && \
    micromamba clean -af --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install cython matscipy prettytable plumed && \
    pip install git+https://github.com/molmod/molmod.git@f59506594b49f7a8545aef0ae6fb378e361eda80 && \
    pip install git+https://github.com/molmod/yaff.git@422570c89e3c44b29db3714a3b8a205279f7b713 && \
    pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install e3nn==0.4.4 && \
    pip install git+https://github.com/mir-group/nequip.git@v0.5.6 && \
    pip install git+https://github.com/mir-group/allegro.git --no-deps && \
    pip install --force git+https://git@github.com/ACEsuit/MACE.git@d520abac437648dafbec0f6e203ec720afa16cf7 --no-deps
RUN pip install git+https://github.com/svandenhaute/psiflow
RUN pip cache purge

USER root
RUN apt remove git gcc g++ -y && apt autoremove -y
USER mambauser
