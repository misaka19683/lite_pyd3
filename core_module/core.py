# ./core_module/core.py
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Parameters:
    """系统参数类，优化了参数验证和计算"""

    def __init__(
            self,
            L: int,
            J_prime: float = 0.0,
            theta: float = 0.0,
            phi: float = 0.0,
            h: float = 0.0,
            boundary: str = "open",
            conserve: Optional[str] = None
    ) -> None:
        """
        初始化系统参数

        Args:
            L: 系统大小
            J_prime: 微扰强度
            theta: 角度θ ∈ [0, π]
            phi: 角度ϕ ∈ [0, 2π]
            h: 磁场强度
            boundary: 边界条件 ('open' or 'periodic')
            conserve: 守恒量 ('Sz', 'parity', 或 None)

        Raises:
            ValueError: 当参数超出有效范围时
        """
        # 参数验证
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L必须是正整数")
        if not (0 <= theta <= np.pi):
            raise ValueError("θ必须在 [0, π] 范围内")
        if not (0 <= phi <= 2 * np.pi):
            raise ValueError("ϕ必须在 [0, 2π] 范围内")
        if boundary not in ["open", "periodic"]:
            raise ValueError("边界条件必须是 'open' 或 'periodic'")
        if conserve not in [None, 'Sz', 'parity']:
            raise ValueError("守恒量必须是 None, 'Sz' 或 'parity'")

        self.L = L
        self.J_prime = J_prime
        self.theta = theta
        self.phi = phi
        self.h = h
        self.boundary = boundary
        self.conserve = conserve

        # 预计算相关参数（避免重复计算）
        self._compute_coupling_constants()

    def _compute_coupling_constants(self) -> None:
        """预计算耦合常数"""
        sin_theta = np.sin(self.theta)
        self.K = sin_theta * np.cos(self.phi)
        self.Gamma = sin_theta * np.sin(self.phi)
        self.J = np.cos(self.theta)

    def __str__(self) -> str:
        return (
            f"Parameters(L={self.L}, J'={self.J_prime}, "
            f"θ={self.theta/np.pi:.2f}π, ϕ={self.phi/np.pi:.2f}π, "
            f"h={self.h}, boundary={self.boundary}, conserve={self.conserve})"
        )

    def __hash__(self) -> int:
        """使参数对象可哈希，用于缓存"""
        return hash((self.L, self.J_prime, self.theta, self.phi,
                     self.h, self.boundary, self.conserve))

    def __eq__(self, other) -> bool:
        """参数比较，用于缓存"""
        if not isinstance(other, Parameters):
            return False
        return (
            self.L == other.L
            and abs(self.J_prime - other.J_prime) < 1e-12
            and abs(self.theta - other.theta) < 1e-12
            and abs(self.phi - other.phi) < 1e-12
            and abs(self.h - other.h) < 1e-12
            and self.boundary == other.boundary
            and self.conserve == other.conserve
        )


class SpinSystemModel(CouplingMPOModel):
    """
    优化的自旋系统模型 - 按哈密顿量分块设计
    每个哈密顿量由公共模块和私有模块拼装而成
    """

    default_lattice = "Chain"
    force_default_lattice = True

    # 数学常量
    SQRT_2 = np.sqrt(2)
    SQRT_3 = np.sqrt(3)
    SQRT_6 = np.sqrt(6)
    SQRT_3_HALF = np.sqrt(3) / 2

    # 分数常量
    ONE_HALF = 1.0 / 2.0
    ONE_THIRD = 1.0 / 3.0
    ONE_SIXTH = 1.0 / 6.0
    TWO_THIRDS = 2.0 / 3.0

    # 阈值常量
    MAGNETIC_FIELD_THRESHOLD = 1e-12
    DELTA2_TERM_THRESHOLD = 1e-8

    def __init__(self, params: Parameters) -> None:
        """
        初始化自旋系统模型
        Args:
            params: 系统参数
        Raises:
            ValueError: 当哈密顿量类型无效时
        """

        self.params = params

        # 构建模型参数
        model_params = {
            "lattice": "Chain",
            "L": params.L,
            "bc_MPS": "infinite" if params.boundary == "periodic" else "finite",
            # "bc_MPS": "finite" # 是否所有的有限系统都应当使用' finite '，这是个需要查阅文档解决的问题
            "bc_x": "periodic" if params.boundary == "periodic" else "open",
            "conserve": params.conserve,
        }
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params: Dict[str, Any]):
        """初始化格点"""
        conserve = model_params.get("conserve", None)
        return SpinHalfSite(conserve=conserve)

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        """初始化哈密顿量项 - 由子类实现"""
        raise NotImplementedError("子类必须实现init_terms方法")

    # ================== 公共磁场模块 ==================

    def _add_magnetic_field_sz(self) -> None:
        """添加Sz方向磁场项 -h*Sz"""
        h = self.params.h
        if abs(h) < self.MAGNETIC_FIELD_THRESHOLD:
            logger.info(f"磁场强度 h={h} 接近零，跳过磁场项")
            return

        for i in range(self.params.L):
            self.add_onsite_term(-h, i, "Sz")

    def _add_magnetic_field_sy_alternating(self) -> None:
        """添加交替Sy方向磁场项 -h*(-1)^i*Sy"""
        h = self.params.h
        if abs(h) < self.MAGNETIC_FIELD_THRESHOLD:
            logger.info(f"磁场强度 h={h} 接近零，跳过磁场项")
            return

        for i in range(self.params.L):
            self.add_onsite_term(-h * (-1) ** i, i, "Sy")

    # ================== 公共耦合模块 ==================

    def _add_coupling_equation16(self, i: int, j: int) -> None:
        """添加方程16耦合项"""
        K, Gamma, J = self.params.K, self.params.Gamma, self.params.J
        J_fix = self.params.J_prime * J
        self._add_coupling_general(Gamma, J, J_fix, K, i, j)

    def _add_coupling_equation20(self, i: int, j: int) -> None:
        """添加方程20耦合项"""
        K, Gamma, J = self.params.K, self.params.Gamma, self.params.J
        self._add_coupling_general(Gamma, J, J, K, i, j)

    def _add_coupling_general(
        self, Gamma: float, J: float, J_fix: float, K: float, i: int, j: int
    ) -> None:
        """通用耦合项添加函数，减少代码重复"""
        r = i % 3
        sign = (-1) ** i
        # XXZ项
        xxz_terms = [
            (Gamma, "Sx", "Sx"),
            (Gamma, "Sy", "Sy"),
            (-Gamma - J, "Sz", "Sz"),
        ]
        for strength, op1, op2 in xxz_terms:
            self.add_coupling_term(strength, i, j, op1, op2)
        # Δ1项
        self._add_delta1_terms(J_fix, r, i, j)
        # Δ2项
        if abs(Gamma - K) > self.DELTA2_TERM_THRESHOLD:
            self._add_delta2_terms(Gamma - K, r, sign, i, j)

    def _add_delta1_terms(self, J_fix: float, r: int, i: int, j: int) -> None:
        """添加Δ1项"""
        if r == 0:
            terms = [(J_fix, "Sx", "Sx"), (-J_fix, "Sy", "Sy")]
        elif r == 1:
            terms = [
                (-self.ONE_HALF * J_fix, "Sx", "Sx"),
                (self.ONE_HALF * J_fix, "Sy", "Sy"),
                (self.SQRT_3_HALF * J_fix, "Sx", "Sy"),
                (self.SQRT_3_HALF * J_fix, "Sy", "Sx"),
            ]
        else:  # r == 2
            terms = [
                (-self.ONE_HALF * J_fix, "Sx", "Sx"),
                (self.ONE_HALF * J_fix, "Sy", "Sy"),
                (-self.SQRT_3_HALF * J_fix, "Sx", "Sy"),
                (-self.SQRT_3_HALF * J_fix, "Sy", "Sx"),
            ]
        for strength, op1, op2 in terms:
            self.add_coupling_term(strength, i, j, op1, op2)

    def _add_delta2_terms(
        self, factor: float, r: int, sign: int, i: int, j: int
    ) -> None:
        """添加Δ2项"""
        if r == 0:
            terms = [
                (factor * (-self.TWO_THIRDS), "Sx", "Sx"),
                (factor * (sign * self.SQRT_2 / 3), "Sx", "Sz"),
                (factor * (-sign * self.SQRT_2 / 3), "Sz", "Sx"),
                (factor * self.ONE_THIRD, "Sz", "Sz"),
            ]
        elif r == 1:
            terms = [
                (factor * (-self.ONE_SIXTH), "Sx", "Sx"),
                (factor * (-1 / (2 * self.SQRT_3)), "Sx", "Sy"),
                (factor * (-1 / (2 * self.SQRT_3)), "Sy", "Sx"),
                (factor * (-self.ONE_HALF), "Sy", "Sy"),
                (factor * (sign / (3 * self.SQRT_2)), "Sz", "Sx"),
                (factor * (-sign / (3 * self.SQRT_2)), "Sx", "Sz"),
                (factor * (sign / self.SQRT_6), "Sz", "Sy"),
                (factor * (-sign / self.SQRT_6), "Sy", "Sz"),
                (factor * self.ONE_THIRD, "Sz", "Sz"),
            ]
        else:  # r == 2
            terms = [
                (factor * (-self.ONE_SIXTH), "Sx", "Sx"),
                (factor * (1 / (2 * self.SQRT_3)), "Sx", "Sy"),
                (factor * (1 / (2 * self.SQRT_3)), "Sy", "Sx"),
                (factor * (-self.ONE_HALF), "Sy", "Sy"),
                (factor * (sign / (3 * self.SQRT_2)), "Sz", "Sx"),
                (factor * (-sign / (3 * self.SQRT_2)), "Sx", "Sz"),
                (factor * (sign / self.SQRT_6), "Sy", "Sz"),
                (factor * (-sign / self.SQRT_6), "Sz", "Sy"),
                (factor * self.ONE_THIRD, "Sz", "Sz"),
            ]
        for strength, op1, op2 in terms:
            self.add_coupling_term(strength, i, j, op1, op2)

    def _get_periodic_term(self,site:int)-> int:
        """获得周期性边界条件下的位点索引"""
        return site%self.lat.N_sites

    def _get_sites_with_boundary(self,sites:List[int])-> Optional[List[int]]:
        """根据边界条件处理位点列表"""
        if self.params.boundary == "periodic":
            sites = [self._get_periodic_term(site) for site in sites]
            return sites
        elif self.params.boundary == "open":
            # 开边界，检查位点是否在有效范围内
            valid_sites=[site for site in sites if 0<=site<self.lat.N_sites]
            if len(valid_sites) != len(sites):
                return None
            return valid_sites
        else:
            raise ValueError(f"Invalid boundary condition:{self.params.boundary}.")

    def _add_multi_terms(self, lambda_: float, start_site: int):
        """添加6位点耦合项

        Args:
            lambda_:耦合强度
            start_site:起始位点索引
        """
        if lambda_ < 1e-10:
            return

        # 基础操作符和位点模式
        ops = ["Sp"] * 6

        # 定义所有的位点组合模式
        site_patterns = [
            [0, 1, 2, 3, 4, 5],  # 连续6个位点
            [0, 1, 2, 3, 4, 6],  # 跳过第5个位点
            [0, 1, 2, 3, 5, 6],  # 跳过第4个位点
            [0, 1, 2, 4, 5, 6],  # 跳过第3个位点
            [0, 1, 3, 4, 5, 6],  # 跳过第2个位点
            [0, 2, 3, 4, 5, 6],  # 跳过第1个位点
            [1, 2, 3, 4, 5, 6],  # 跳过第0个位点
        ]

        # 添加耦合项
        for pattern_idx,pattern in enumerate(site_patterns):
            #计算实际位点索引
            actual_sites=[start_site+offset for offset in pattern]

            # 处理边界条件
            processed_sites=self._get_sites_with_boundary(actual_sites)

            # 如果是开边界并且位点超出范围，跳过此项
            if processed_sites is None:
                continue
            # 计算强度（交替符号）
            strength=lambda_*(-1)**pattern_idx
            sorted_sites,sorted_ops=self._sort_sites_and_ops(processed_sites,ops)
            try:
                self.add_multi_coupling_term(
                    strength=strength,ijkl=sorted_sites,ops_ijkl=sorted_ops,op_string=['Id']*5
                )
            except Exception as e:
                logger.warning(f"Error occurred when adding multi-term: {e},sites={processed_sites}")

    def _add_multi_terms_2(self,lambda_:float,start_site:int)->None:
        """添加复杂的多位点耦合项
        式子20的第二个多点耦合项
        Args:
            lambda_:耦合强度
            start_site:起始位点索引
        """
        if lambda_<1e-10:
            return

        # 定义不同类型的耦合项
        coupling_configs = [
            {
                "ops": ["Sm"] * 6,
                "patterns": [
                    ([0, 1, 2, 3, 4, 5],+1)
                ],
            },
            {
                "ops": ["Sm"] * 5 + ["Sp"],
                "patterns": [
                    ([0, 1, 2, 3, 4, 5],-1),  # -
                    # ([0, 2, 3, 4, 5, 3],+1),  # +
                    # ([1, 2, 3, 4, 5, 3],-1),  # -
                ],
            },
            # 对上面的重复算符做出修正
            #########################################
            {
                "ops": ["Sm"] * 4 + ["Sz"],
                "patterns": [
                    ([0, 2, 4, 5, 3],+1),  # +
                    ([1, 2, 4, 5, 3],-1),  # -
                ]
            },
            {
                "ops": ["Sm"] * 4,
                "patterns": [
                    ([0, 2, 4, 5],+1/2),  # +
                    ([1, 2, 4, 5],-1/2),  # -
                ]
            },
            ###########################################

            {
                "ops": ["Sm"] * 5 + ["Sp"],
                "patterns": [
                    ([0, 1, 2, 3, 4, 6],-1),  # -
                    ([0, 1, 2, 3, 5, 6],+1),  # +
                    ([0, 1, 2, 4, 5, 6],-1),  # -
                ],
            },
            {
                "ops": ["Sm"] * 4 + ["Sp"] * 2,
                "patterns": [
                    # ([0, 1, 3, 4, 3, 6],+1),  # +
                    # ([0, 2, 3, 4, 3, 6],-1),  # -
                    # ([1, 2, 3, 4, 3, 6],+1),  # +
                    # ([0, 1, 3, 5, 3, 6],-1),  # -
                    # ([0, 2, 3, 5, 3, 6],+1),  # +
                    # ([1, 2, 3, 5, 3, 6],-1),  # -
                    ([0, 1, 4, 5, 3, 6],+1),  # +
                    ([0, 2, 4, 5, 3, 6],-1),  # -
                    ([1, 2, 4, 5, 3, 6],+1),  # +
                ],
            },
            # 对上面的重复算符做出修正
            #########################################
            {
                "ops": ["Sm"] * 3 + ["Sz","Sp"],
                "patterns": [
                    ([0, 1, 4, 3, 6],+1),  # +
                    ([0, 2, 4, 3, 6],-1),  # -
                    ([1, 2, 4, 3, 6],+1),  # +
                    ([0, 1, 5, 3, 6],-1),  # -
                    ([0, 2, 5, 3, 6],+1),  # +
                    ([1, 2, 5, 3, 6],-1),  # -
                ]
            },
            {
                "ops": ["Sm"] * 3 + ["Sp"],
                "patterns": [
                    ([0, 1, 4, 6],+1/2),  # +
                    ([0, 2, 4, 6],-1/2),  # -
                    ([1, 2, 4, 6],+1/2),  # +
                    ([0, 1, 5, 6],-1/2),  # -
                    ([0, 2, 5, 6],+1/2),  # +
                    ([1, 2, 5, 6],-1/2),  # -
                ]
            },
            ########################################
        ]

        for config in coupling_configs:
            ops = config["ops"]

            for pattern,sign in config["patterns"]:
                # print(pattern)
                # 计算实际位点索引
                actual_sites = [start_site + int(offset) for offset in pattern]

                # 处理边界条件
                processed_sites = self._get_sites_with_boundary(actual_sites)

                if processed_sites is None:
                    continue

                # 计算强度
                strength = lambda_ * sign
                sorted_sites,sorted_ops=self._sort_sites_and_ops(processed_sites,ops)
                print(f"[DEBUG] pattern={pattern}, sites={sorted_sites}")
                try:
                    self.add_multi_coupling_term(
                        strength=strength,
                        ijkl=sorted_sites,
                        ops_ijkl=sorted_ops,
                        op_string= ["Id"] * (len(ops) - 1)

                    )
                except Exception as e:
                    logger.warning(
                        f"添加复杂多位点耦合项失败: sites={processed_sites},ops={ops} error={e}"
                    )

    def add_multi_terms(self, lambda_: float):
        if abs(lambda_) < 1e-10:
            return

        if self.params.boundary == "periodic":
            for i in range(self.lat.N_sites):
                self._add_multi_terms(lambda_, i)
                self._add_multi_terms_2(lambda_, i)
        elif self.params.boundary == "open":
            for i in range(self.lat.N_sites):
                self._add_multi_terms(lambda_, i)
                self._add_multi_terms_2(lambda_, i)
        else:
            raise ValueError(f"Invalid boundary condition:{self.params.boundary}.")
# term1=(
#     + s0+*s1+*s2+*s3+*s4+*s5+
#     - s0+*s1+*s2+*s3+*s4+*s6+
#     + s0+*s1+*s2+*s3+*s5+*s6+
#     - s0+*s1+*s2+*s4+*s5+*s6+
#     + s0+*s1+*s3+*s4+*s5+*s6+
#     - s0+*s2+*s3+*s4+*s5+*s6+
#     + s1+*s2+*s3+*s4+*s5+*s6+
#        )
#     term2=(
#     + s0-*s1-*s2-*s3-*s4-*s5-
#     - s0-*s1-*s2-*s3-*s4-*s6-
#     + s0-*s1-*s2-*s3-*s5-*s6-
#     - s0-*s1-*s2-*s4-*s5-*s6-
#     + s0-*s1-*s3-*s4-*s5-*s6-
#     - s0-*s2-*s3-*s4-*s5-*s6-
#     + s1-*s2-*s3-*s4-*s5-*s6-
#     )
    def add_multi_terms_new(self,lambda_:float):
        if abs(lambda_) < 1e-10:
            return

        sps=['Sp']*6 # S+
        sms=['Sm']*6 # S-

        site_patterns = (
            [0, 1, 2, 3, 4, 5],  # 跳过位点6
            [0, 1, 2, 3, 4, 6],  # 跳过位点5
            [0, 1, 2, 3, 5, 6],  # 跳过位点4
            [0, 1, 2, 4, 5, 6],  # 跳过位点3
            [0, 1, 3, 4, 5, 6],  # 跳过位点2
            [0, 2, 3, 4, 5, 6],  # 跳过位点1
            [1, 2, 3, 4, 5, 6],  # 跳过位点0
        )

        site_signal = [1, -1, 1, -1, 1, -1, 1]

        for i in range(self.lat.N_sites):
            for idx, pattern in enumerate(site_patterns):
                actual_site = [i + offset for offset in pattern]
                processing_site = self._get_sites_with_boundary(actual_site)
                if processing_site is None:
                    continue
                strength=lambda_*site_signal[idx]
                sorted_sites,sorted_ops=self._sort_sites_and_ops(processing_site,sps)
                try:
                    self.add_multi_coupling_term(
                        strength=strength,
                        ijkl=sorted_sites,
                        ops_ijkl=sorted_ops,
                        op_string=['Id']*5
                    )
                except Exception as e:
                    logger.warning(f"Error occurred when adding multi-term: {e},sites={processing_site}")

                sorted_sites,sorted_ops=self._sort_sites_and_ops(processing_site,sms)
                try:
                    self.add_multi_coupling_term(
                        strength=strength,
                        ijkl=sorted_sites,
                        ops_ijkl=sorted_ops,
                        op_string=['Id']*5
                    )
                except Exception as e:
                    logger.warning(f"Error occurred when adding multi-term: {e},sites={processing_site}")

        

    @staticmethod
    def _sort_sites_and_ops(
            sites: List[int], ops: List[str]
    ) -> Tuple[List[int], List[str]]:
        """同时对站点索引和算符进行排序，保持对应关系

        Args:
            sites: 站点索引列表
            ops: 对应的算符列表

        Returns:
            排序后的 (sites, ops) 元组
        """
        if len(sites) != len(ops):
            raise ValueError(f"站点数量({len(sites)})与算符数量({len(ops)})不匹配")

        # 创建索引-站点-算符的组合，然后按站点索引排序
        combined = list(zip(sites, ops))
        combined_sorted = sorted(combined, key=lambda x: x[0])

        # 分离排序后的站点和算符
        sorted_sites, sorted_ops = zip(*combined_sorted) if combined_sorted else ([], [])

        return list(sorted_sites), list(sorted_ops)



class Equation1Hamiltonian(SpinSystemModel):
    """构建原始哈密顿量"""

    def _add_coupling_original(self, i: int, j: int) -> None:
        """添加原始耦合项"""
        K, Gamma, J = self.params.K, self.params.Gamma, self.params.J
        if i % 2 == 0:  # 偶数位点
            couplings = [
                (K, "Sy", "Sy"),
                (Gamma, "Sz", "Sx"),
                (Gamma, "Sx", "Sz"),
            ]
        else:  # 奇数位点
            couplings = [
                (K, "Sx", "Sx"),
                (Gamma, "Sy", "Sz"),
                (Gamma, "Sz", "Sy"),
            ]
        # 添加特定耦合
        for strength, op1, op2 in couplings:
            self.add_coupling_term(strength, i, j, op1, op2)
        # 添加各向同性交换项
        for op in ["Sx", "Sy", "Sz"]:
            self.add_coupling_term(J, i, j, op, op)

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 最近邻耦合
        for i in range(L - 1):
            self._add_coupling_original(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_original(0, L - 1)


class Equation4Hamiltonian(SpinSystemModel):
    """构建六角旋转哈密顿量"""

    def _add_coupling_six_rotated(self, i: int, j: int) -> None:
        """添加六角旋转耦合项"""
        K, Gamma, J = self.params.K, self.params.Gamma, self.params.J
        r = i % 3
        # 使用查找表简化逻辑
        coupling_patterns = {
            0: [
                ("Sy", "Sy", -K),
                ("Sz", "Sz", -Gamma),
                ("Sx", "Sx", -Gamma),
                ("Sy", "Sy", -J),
                ("Sz", "Sx", -J),
                ("Sx", "Sz", -J),
            ],
            1: [
                ("Sx", "Sx", -K),
                ("Sy", "Sy", -Gamma),
                ("Sz", "Sz", -Gamma),
                ("Sx", "Sx", -J),
                ("Sy", "Sz", -J),
                ("Sz", "Sy", -J),
            ],
            2: [
                ("Sz", "Sz", -K),
                ("Sx", "Sx", -Gamma),
                ("Sy", "Sy", -Gamma),
                ("Sz", "Sz", -J),
                ("Sx", "Sy", -J),
                ("Sy", "Sx", -J),
            ],
        }
        for op1, op2, strength in coupling_patterns[r]:
            self.add_coupling_term(strength, i, j, op1, op2)

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 最近邻耦合
        for i in range(L - 1):
            self._add_coupling_six_rotated(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_six_rotated(0, L - 1)


class Equation6Hamiltonian(SpinSystemModel):
    """构建六角二次旋转哈密顿量"""

    def _add_coupling_six_two_rotated(self, i: int, j: int) -> None:
        """添加六角二次旋转耦合项"""
        K, Gamma, J = self.params.K, self.params.Gamma, self.params.J
        self._add_coupling_general(Gamma, J, J, K, i, j)

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 最近邻耦合
        for i in range(L - 1):
            self._add_coupling_six_two_rotated(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_six_two_rotated(0, L - 1)


class Equation16Hamiltonian(SpinSystemModel):
    """构建方程16哈密顿量"""

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 最近邻耦合
        for i in range(L - 1):
            self._add_coupling_equation16(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation16(0, L - 1)


class Equation18Hamiltonian(SpinSystemModel):
    """构建磁场方程18哈密顿量"""

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 基础耦合项（与equation16相同）
        for i in range(L - 1):
            self._add_coupling_equation16(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation16(0, L - 1)

        # 磁场项 -h*Sz
        self._add_magnetic_field_sz()


class Equation19Hamiltonian(SpinSystemModel):
    """构建磁场方程19哈密顿量"""

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 基础耦合项（与equation16相同）
        for i in range(L - 1):
            self._add_coupling_equation16(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation16(0, L - 1)

        # 交替磁场项 -h*(-1)^i*Sy
        self._add_magnetic_field_sy_alternating()


class Equation20Hamiltonian(SpinSystemModel):
    """构建方程20哈密顿量"""

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 最近邻耦合
        for i in range(L - 1):
            self._add_coupling_equation20(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation20(0, L - 1)
        # 复用在此函数中没有用的J_prime作为lambda的数值
        # self.add_multi_terms(self.params.J_prime)
        self.add_multi_terms_new(self.params.J_prime)

class Equation22Hamiltonian(SpinSystemModel):
    """构建磁场方程22哈密顿量"""

    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L = self.params.L
        # 基础耦合项（与equation20相同）
        for i in range(L - 1):
            self._add_coupling_equation20(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation20(0, L - 1)

        # 磁场项 -h*Sz
        self._add_magnetic_field_sz()
        # 复用在此函数中没有用的J_prime作为lambda的数值
        # self.add_multi_terms(self.params.J_prime)
        self.add_multi_terms_new(self.params.J_prime)

class Equation23Hamiltonian(SpinSystemModel):
    """构建磁场方程23哈密顿量"""
    def init_terms(self, model_params: Dict[str, Any]) -> None:
        # super().init_terms(model_params)
        L=self.params.L
        # 基础耦合项（与equation20相同）
        for i in range(L - 1):
            self._add_coupling_equation20(i, i + 1)

        # 周期性边界条件
        if self.params.boundary == "periodic":
            self._add_coupling_equation20(0, L - 1)

        # 交替磁场项 -h*(-1)^i*Sy
        self._add_magnetic_field_sy_alternating()
        # 复用在此函数中没有用的J_prime作为lambda的数值
        # self.add_multi_terms(self.params.J_prime)
        self.add_multi_terms_new(self.params.J_prime)





# ====================== 优化的DMRG工具函数 ======================

@lru_cache(maxsize=128)
def get_optimized_dmrg_params(system_size: int, precision: str = "normal") -> Dict[str, Any]:
    """
    获取优化的DMRG参数，基于系统大小自适应调整

    Args:
        system_size: 系统大小
        precision: 精度级别 ('fast', 'normal', 'high')

    Returns:
        优化的DMRG参数字典
    """

    match precision:
        case "fase":
            return {
                "trunc_params": {
                    "chi_max": min(50, system_size * 2),
                    "svd_min": 1e-6,
                    "trunc_cut": 1e-8,
                },
                "mixer": True,
                "max_sweeps": 10,
                "chi_list": {0: 20, 3: min(50, system_size * 2)},
                "max_E_err": 1e-8,
                "max_S_err": 1e-6,
            }
        case "high":
            return {
                "trunc_params": {
                    "chi_max": min(200, system_size * 4),
                    "svd_min": 1e-10,
                    "trunc_cut": 1e-12,
                },
                "mixer": True,
                "max_sweeps": 25,
                "chi_list": {
                    0: 20,
                    5: 50,
                    10: min(100, system_size * 2),
                    15: min(200, system_size * 4),
                },
                "max_E_err": 1e-12,
                "max_S_err": 1e-10,
            }
        case "1000_10":
            return {
                "trunc_params": {
                    "chi_max": 1000,
                    "svd_min": 1e-8,
                    "trunc_cut": 1e-10,
                },
                "mixer": True,
                "max_sweeps": 100,
                "chi_list": {
                    0: 20,  # 初始快速收敛
                    3: 40,  # 早期加速
                    6: 80,  # 中期增长
                    10: 200,  # 稳定增长
                    15: 500,  # 高精度准备
                    20: 1000,  # 最终精度
                },
                "max_E_err": 1e-12,
                "max_S_err": 1e-10,
            }
        case "1000_12":
            return {
                "trunc_params": {
                    "chi_max": 1000,
                    "svd_min": 1e-10,
                    "trunc_cut": 1e-12,
                },
                "max_sweeps": 100,
                "chi_list": {
                    0: 20,  # 初始快速收敛
                    3: 40,  # 早期加速
                    6: 80,  # 中期增长
                    10: 200,  # 稳定增长
                    15: 500,  # 高精度准备
                    20: 1000,  # 最终精度
                },
                "max_E_err": 1e-12,
                "max_S_err": 1e-10,
            }
        case "supercomputer":
            return {
                "trunc_params": {
                    "chi_max": 1000,
                    "svd_min": 1e-7,
                    "trunc_cut": 1e-9,
                },
                "mixer": True,
                "max_sweeps": 80,  # 减少最大扫描次数
                "chi_list": {
                    0: 10,        # 更小的初始值，快速warm-up
                    8: 20,        # 更早开始增长
                    12: 40,        #
                    16: 80,        #
                    20: 160,       # 更密集的增长
                    22: 200,
                    24: 240,
                    26: 280,
                    28: 320,      #
                    30: 480,
                    32: 640,      #aa
                    34: 800,
                    36: 960,
                    38: 1000,     # 最终精度
                },
                "max_E_err": 1e-11,  # 略微放宽能量收敛
                "max_S_err": 1e-9,   # 略微放宽熵收敛

            }
        case _: # normal
            return {
                "trunc_params": {
                    "chi_max": min(100, system_size * 3),
                    "svd_min": 1e-8,
                    "trunc_cut": 1e-10,
                },
                "mixer": True,
                "max_sweeps": 15,
                "chi_list": {0: 20, 5: 50, 10: min(100, system_size * 3)},
                "max_E_err": 1e-10,
                "max_S_err": 1e-8,
            }


def run_dmrg_optimized(model: SpinSystemModel,
                      dmrg_params: Optional[Dict[str, Any]] = None,
                      precision: str = "normal") -> Tuple[float, MPS]:
    """
    优化的DMRG计算函数

    Args:
        model: 自旋系统模型
        dmrg_params: DMRG参数（可选）
        precision: 精度级别

    Returns:
        (energy, psi): 基态能量和波函数

    Raises:
        ValueError: 当模型无效时
    """
    if model is None or not hasattr(model, "lat"):
        raise ValueError("无效的模型")

    # 使用优化的参数
    if dmrg_params is None:
        dmrg_params = get_optimized_dmrg_params(model.lat.N_sites, precision)

    sites = model.lat.mps_sites()

    # 创建更好的初始态
    psi = create_initial_state(sites, model.lat.bc_MPS, model.params)

    # 运行DMRG
    try:
        eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        energy, psi_optimized = eng.run()

        logger.info(f"DMRG收敛，能量: {energy:.10f}")
        return energy, psi_optimized

    except Exception as ee:
        logger.error(f"DMRG计算失败: {ee}")
        raise


def create_initial_state(sites, bc_MPS: str, params: Parameters) -> MPS:
    """
    创建优化的初始态

    Args:
        sites: 格点列表
        bc_MPS: 边界条件
        params: 系统参数

    Returns:
        初始MPS态
    """
    L = len(sites)

    if params.conserve == "Sz":
        # 创建总自旋为零的态
        N_up = L // 2
        initial_states = ["up"] * N_up + ["down"] * (L - N_up)
    else:
        # 创建交替自旋态
        initial_states = []
        for i in range(L):
            # 添加一些随机性避免局部最小值
            if i % 4 in [0, 3]:
                initial_states.append("up")
            else:
                initial_states.append("down")

    psi = MPS.from_product_state(sites, initial_states, bc=bc_MPS)
    psi.canonical_form()
    return psi


def compute_bond_energies_optimized(model: SpinSystemModel, psi: MPS) -> np.ndarray:
    """
    优化的键能计算函数

    Args:
        model: 模型
        psi: 波函数

    Returns:
        键能数组
    """
    L = model.lat.N_sites

    try:
        # 尝试使用最优方法
        from tenpy.models.model import NearestNeighborModel
        nn_model = NearestNeighborModel.from_MPOModel(model)
        bond_energies = nn_model.bond_energies(psi)
        logger.info("使用 NearestNeighborModel 成功计算键能")
        return bond_energies

    except Exception as ee:
        logger.warning(f"NearestNeighborModel 方法失败: {ee}")

        # 备选方案：手动计算
        try:
            H_bonds = model.calc_H_bond(tol_zero=1e-15)
            bond_energies = np.zeros(L - 1)

            for i in range(min(len(H_bonds), L - 1)):
                if H_bonds[i] is not None:
                    bond_energies[i] = np.real(psi.expectation_value([H_bonds[i]]))

            logger.info("使用备选方案成功计算键能")
            return bond_energies

        except Exception as e2:
            logger.warning(f"备选方案失败: {e2}")

            # 最终备选：均匀分布
            try:
                total_energy = np.real(psi.expectation_value(model.H_MPO))
                avg_energy = total_energy / (L - 1)
                bond_energies = np.full(L - 1, avg_energy)
                logger.warning(f"使用平均能量分布: {avg_energy:.6f}")
                return bond_energies
            except (AttributeError, TypeError, ZeroDivisionError) as e3:
                logger.error(f"所有键能计算方法都失败: {e3}")
                raise RuntimeError(f"无法计算键能: {e3}") from e3



def compute_energy_second_derivative(bond_energies: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算能量密度的二阶导数，优化了数值稳定性

    Args:
        bond_energies: 键能数组
        N: 系统大小

    Returns:
        (Ea, x): 二阶导数和对应的x值
    """
    bond_energies = np.asarray(bond_energies)

    if len(bond_energies) < 3:
        logger.warning("键能数组太短，无法计算二阶导数")
        return np.array([]), np.array([])

    # 计算二阶差分
    second_diff = np.diff(bond_energies, n=2) / 4.0

    # 对应的索引
    r_values = np.arange(1, len(bond_energies) - 1)

    # 应用交替符号
    Ea = second_diff * (-1) ** r_values

    # 计算x值
    x = np.sin(np.pi * r_values / N) * N / np.pi

    logger.info(f"计算二阶导数: {len(Ea)} 个数据点")
    return Ea, x


# ====================== 优化的问题求解函数 ======================


def solve_single_hamiltonian(params:Parameters,model:SpinSystemModel):

    try:
        logger.info(f"开始计算{model}")
        energy,_=run_dmrg_optimized(model,precision="normal")
        logger.info(f"✓ {model} 完成，能量: {energy:.10f}")
        return energy
    except Exception as err:
        logger.error(f"✗ {model} 计算失败: {err}")
        raise


def compute_bond_energies(model, psi):
    """计算键能"""
    L = model.lat.N_sites
    bond_energies = np.zeros(L - 1)

    # 计算每个键的能量
    for i in range(L - 1):
        # 创建仅包含当前键的临时模型
        temp_params = model.params
        bond_model = type(model)(temp_params)

        # 只添加当前键 (i, i+1) 的耦合
        if hasattr(model, "_add_coupling_original"):
            # 使用模型的内部方法添加单个键的耦合
            bond_model._add_coupling_original(i, i + 1)
        else:
            # 手动添加当前键的耦合项
            # 这里需要根据具体的哈密顿量类型来添加相应的耦合项
            # 以原始哈密顿量为例：
            K, Gamma, J = temp_params.K, temp_params.Gamma, temp_params.J
            if i % 2 == 0:  # 偶数位点
                couplings = [
                    (K, "Sy", "Sy"),
                    (Gamma, "Sz", "Sx"),
                    (Gamma, "Sx", "Sz"),
                ]
            else:  # 奇数位点
                couplings = [
                    (K, "Sx", "Sx"),
                    (Gamma, "Sy", "Sz"),
                    (Gamma, "Sz", "Sy"),
                ]

            # 添加特定耦合
            for strength, op1, op2 in couplings:
                bond_model.add_coupling_term(strength, i, i + 1, op1, op2)

            # 添加各向同性交换项
            for op in ["Sx", "Sy", "Sz"]:
                bond_model.add_coupling_term(J, i, i + 1, op, op)

        # 计算键能
        bond_energy = np.real(psi.expectation_value(bond_model.H_MPO))
        bond_energies[i] = bond_energy

    return bond_energies





