from dataclasses import dataclass, fields
from typing import ClassVar

__all__ = ["SlurmHeader", "PythonCall", "SrunCall", "SingularityCall", "generate_submission_script"]

@dataclass(frozen=True, kw_only=True)
class SlurmHeader:
    """
    SBATCH header data for a slurm submission script.
    The __str__ method will generate the script header.

    See slurm documentation for information on each attribute.

    Slurm `output` and `error` files will be created in `job_directory`.
    The bash executable can be specified with `bash_path`.
    """

    # These options will only be included if their value is set
    optionals: ClassVar = ("switches", "distribution", "hint",
                           "mail_user", "mail_type")
    # These options are only included if True
    flags: ClassVar = ( "exclusive", "requeue")

    # sbatch 
    account: str
    partition: str = "standard"
    qos: str = "standard"
    job_name: str
    output: str = "slurm-%x-%j.out"
    error: str = "slurm-%x-%j.out"
    nodes: int
    ntasks_per_node: int
    time: tuple((int, int, int))
    switches: int = -1
    distribution: str = ""
    hint: str = ""
    exclusive: bool = False
    requeue: bool = False
    mail_user: str = ""
    mail_type: str = ""
    bash_path: str = "!/bin/bash"
    job_directory: str = ""

    def __str__(self):
        job_directory = self.job_directory
        if job_directory[-1] == "/":
            job_directory = job_directory[:-1]
        header = \
f"""#{self.bash_path}
#
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --qos={self.qos}
#
#SBATCH --job-name={self.job_name}
#SBATCH --output={job_directory}/{self.output}
#SBATCH --error={job_directory}/{self.error}
#
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.ntasks_per_node}
#
#SBATCH --time={':'.join(map(lambda t: f'{t:02}', self.time))}
"""
        # options which should only be included if set
        all_fields = fields(self)
        for option in self.optionals:
            value = getattr(self, option)
            for field in all_fields:
                if field.name == option:
                    default = field.default
                    break
            if value != default:
                header += \
f"""#
#SBATCH --{option.replace('_','-')}={value}
"""
        # options wichh are set by their presence, not value
        for flag in self.flags:
            if getattr(self, flag):
                header += \
f"""#
#SBATCH --{flag}
"""
        return header

@dataclass(frozen=True, kw_only=True)
class PythonCall:
    path: str = "python"
    args: str = ""
    script_name: str
    script_dir: str
    script_args: str = ""


@dataclass(frozen=True, kw_only=True)
class SrunCall:
    l3cores: int = -1
    l3size: int = -1
    nodesize: int = -1
    args: str = ""
    xthi: bool = False

    def cpu_map_cmd(self, l3size="${L3SIZE}", l3cores="${L3CORES}", nodesize="${NODESIZE}"):
        return f"""$(python3 -c "print(','.join(map(str,filter(lambda i: (i%{l3size})<{l3cores}, range({nodesize})))))")"""

    def __str__(self):
        string = ""
        args = f"{self.args}"

        # do we set the cpu_map?
        cpu_map_vals = (self.l3cores, self.l3size, self.nodesize)
        if any(val > 0 for val in cpu_map_vals):
            if not all(val > 0 for val in cpu_map_vals):
                msg = "Must set all of l3cores,l3size,nodesize to generate an srun cpu_map"
                raise ValueError(msg)
            else:
                args += " --cpu_bind=map_cpu:${CPU_MAP}"
                string += \
f"""
# L3 cache is the lowest level of shared memory, with L3SIZE cores per cache
# and NODESIZE/L3SIZE L3 caches per node. Using fewer than L3SIZE cores/L3
# cache may improve strong scaling performance for memory bound applications.
# The ntasks-per-node value must be equal to L3CORES*(NODESIZE/L3SIZE).

L3CORES={self.l3cores}
L3SIZE={self.l3size}
NODESIZE={self.nodesize}

CPU_MAP={self.cpu_map_cmd()}
"""

        # variable for the full srun call
        string += \
f"""
SRUN_ARGS="{args}"
SRUN_CALL="srun ${{SRUN_ARGS}}"
"""
        return string


@dataclass(frozen=True, kw_only=True)
class SingularityCall:
    container: str
    args: str = ""
    bind_from: str = ""
    bind_to: str = ""
    home: str = ""
    directory: str = ""
    setup_file: str = ""

    def full_args(self):
        if self.home != "":
            args = f"--home {self.home}"
        if self.bind_from != "":
            if (args != "") and (not args.endswith(" ")):
                args += " "
            args += f"--bind {self.bind_from}"
        if self.bind_to != "":
            args += f":{self.bind_to}"
        return args


def generate_submission_script(header, python, srun, singularity, script_name=None):
    output_file = f"{header.job_directory}/{header.output}"
    output_file = output_file.replace("%x", "${SLURM_JOB_NAME}")
    output_file = output_file.replace("%j", "${SLURM_JOB_ID}")

    error_file = f"{header.job_directory}/{header.error}"
    error_file = error_file.replace("%x", "${SLURM_JOB_NAME}")
    error_file = error_file.replace("%j", "${SLURM_JOB_ID}")

    script = str(header)
    script += \
f"""
# exit on first error
set -e

# print commands
set -x

### === --- Unique identifier and directory

JOBCODE=${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}
JOBDIR={header.job_directory}/${{JOBCODE}}
mkdir -p ${{JOBDIR}}

### === --- Python scripts and arguments

PYTHON_SCRIPT={python.script_name}
PYTHON_SCRIPT_DIR={python.script_dir}

PYTHON_SCRIPT_ARGS="{python.script_args}"

### === --- Python executable

PYTHON_ARGS="{python.args}"
PYTHON_EXEC="{python.path}"

PYTHON_CALL="${{PYTHON_EXEC}} ${{PYTHON_ARGS}}"

### === --- Setup the environment for singularity

export SIFDIR="{singularity.directory}"
CONTAINER="{singularity.container}"

SINGULARITY_ARGS="{singularity.full_args()}"
SINGULARITY_CALL="singularity run ${{SINGULARITY_ARGS}}"

source ${{SIFDIR}}/{singularity.setup_file}

### === --- srun call

{srun}

### === --- Print rank layout or not?

RUN_XTHI={int(srun.xthi)}

### === --- ------------------------------------------------ --- === ###
### === --- Usually won't need to change anything below here --- === ###
### === --- ------------------------------------------------ --- === ###

### === --- Some job info for debugging

set +x
echo -e "Job started at " `date`
echo ""

echo "What job is running?"
echo SLURM_JOB_ID          = $SLURM_JOB_ID
echo SLURM_JOB_NAME        = $SLURM_JOB_NAME
echo SLURM_JOB_ACCOUNT     = $SLURM_JOB_ACCOUNT
echo ""

echo "Where is the job running?"
echo SLURM_CLUSTER_NAME    = $SLURM_CLUSTER_NAME
echo SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION
echo SLURM_JOB_QOS         = $SLURM_JOB_QOS
echo SLURM_SUBMIT_DIR      = $SLURM_SUBMIT_DIR
echo ""

echo "What are we running on?"
echo SLURM_DISTRIBUTION    = $SLURM_DISTRIBUTION
echo SLURM_NTASKS          = $SLURM_NTASKS
echo SLURM_NTASKS_PER_NODE = $SLURM_NTASKS_PER_NODE
echo SLURM_JOB_NUM_NODES   = $SLURM_JOB_NUM_NODES
echo SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST
echo ""
set -x

### === --- Check the rank layout

if [[ ${{RUN_XTHI}} -gt 0 ]]; then
   export MPICH_ENV_DISPLAY=1
   set +x
   echo module load xthi
   module load xthi
   set -x
   echo -e "xthi started at " `date`
   $SRUN_CALL xthi > ${{JOBDIR}}/xthi.log 2>&1
   echo -e "xthi finished at " `date`
   unset MPICH_ENV_DISPLAY
fi

### === --- Copy files to nodes

echo -e "Start copying files to nodes: " `date`
echo -e ""

TMP_SCRIPT=${{JOBDIR}}/${{JOBCODE}}-${{PYTHON_SCRIPT}}

cp ${{PYTHON_SCRIPT_DIR}}/${{PYTHON_SCRIPT}} ${{TMP_SCRIPT}}

sbcast --compress=none ${{SIFDIR}}/${{CONTAINER}} /tmp/${{CONTAINER}}

### === --- Run the script

set +x
echo -e "Script start time: " `date`
echo -e ""

set -x
${{SRUN_CALL}} ${{SINGULARITY_CALL}} /tmp/${{CONTAINER}} \ 
    ${{PYTHON_CALL}} ${{TMP_SCRIPT}} ${{PYTHON_SCRIPT_ARGS}}

echo -e "Script end time: " `date`

OUTPUT_FILE={output_file}
cp ${{OUTPUT_FILE}} ${{JOBDIR}}/${{OUTPUT_FILE}}
"""
    if error_file != output_file:
        script += \
f"""
ERROR_FILE={error_file}
cp ${{ERROR_FILE}} ${{JOBDIR}}/${{ERROR_FILE}}
"""

    if script_name:
        with open(script_name, "w") as f:
            f.write(script)

    return script


if __name__ == "__main__":

    common_slurm_args = {
        'account': 'e781',
        'partition': 'standard',
        'qos': 'standard',
        'hint': "nomultithread",
        'distribution': "block:block",
        'exclusive': True,
        'requeue': True,
    }

    header = SlurmHeader(
        job_name='test_job',
        job_directory="results",
        nodes=1,
        ntasks_per_node=2,
        time=(0,5,0),
        **common_slurm_args
    )

    python = PythonCall(
        script_name="script.py",
        script_dir="examples",
        script_args="--metrics_dir=${JOBDIR} --extra_arg=value",
        path="/home/firedrake/firedrake/bin/python",
    )

    srun = SrunCall(
        l3cores=2,
        l3size=4,
        nodesize=128,
        args="--hint=nomultithread",
        xthi=True,
    )

    singularity = SingularityCall(
        directory="/work/e781/shared/firedrake-singularity",
        container="firedrake-archer2.sif",
        bind_from="$PWD",
        bind_to="/home/firedrake/work",
        home="$PWD",
        setup_file="singularity_setup.sh",
    )

    script = generate_submission_script(
        header, python, srun, singularity, script_name="test-jobscript.sh")

    print(script)
