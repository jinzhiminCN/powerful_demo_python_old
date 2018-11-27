# -*- coding:utf-8 -*-

# ==============================================================================
# 测试sk的相关特征工程方法。
# 参考：https://www.cnblogs.com/jasonfreak/p/5448385.html
# IRIS数据集由Fisher在1936年整理，包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、
# Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），特征值都为正浮点数，单位为厘米。目标值为鸢尾花
# 的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））
# ==============================================================================
from numpy import vstack, array, nan, log1p
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, \
    Binarizer, OneHotEncoder, Imputer, PolynomialFeatures, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from minepy import MINE
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# 导入IRIS数据集
iris = load_iris()


def log_print_value(name, value):
    """
    使用日志打印numpy的运算结果。
    :param name
    :param value
    :return:
    """
    common_logger.info("{0}:\n{1}".format(name, value))


def show_data():
    """
    显示数据。
    :return:
    """
    # 特征矩阵
    feature_matrix = iris.data
    log_print_value("feature_matrix:", feature_matrix)

    # 目标向量
    target = iris.target
    log_print_value("target:", target)


def test_nondimensionalization():
    """
    测试无量纲化。
    无量纲化使不同规格的数据转换到同一规格。常见的无量纲化方法有标准化和区间缩放法。
    标准化：标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。
    区间缩放法：区间缩放法利用边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0, 1]等。
    标准化：标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
    归一化：归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
    拥有统一的标准，也就是说都转化为“单位向量”。
    :return:
    """
    # 标准化，返回值为标准化后的数据
    standard_data = StandardScaler().fit_transform(iris.data)
    log_print_value("standard_data:", standard_data)

    # 区间缩放，返回值为缩放到[0, 1]区间的数据
    min_max_scale_data = MinMaxScaler().fit_transform(iris.data)
    log_print_value("min_max_scale_data:", min_max_scale_data)

    # 归一化，返回值为归一化后的数据
    normalize_data = Normalizer().fit_transform(iris.data)
    log_print_value("normalize_data:", normalize_data)


def test_binarize():
    """
    测试数据二值化运算。
    :return:
    """
    # 二值化，阈值设置为3，返回值为二值化后的数据
    binarize_data = Binarizer(threshold=3).fit_transform(iris.data)
    log_print_value("binarize_data:", binarize_data)


def test_one_hot():
    """
    测试数据的one hot编码。对定性特征哑编码。
    :return:
    """
    # 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
    one_hot_data = OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))
    log_print_value("one_hot_data:", one_hot_data)


def test_missing_value():
    """
    测试缺失值的补充。
    由于IRIS数据集没有缺失值，故对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失。
    :return:
    """
    # 缺失值计算，返回值为计算缺失值后的数据
    # 参数missing_value为缺失值的表示形式，默认为NaN
    # 参数strategy为缺失值填充方式，默认为mean（均值）
    all_data = Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
    log_print_value("all_data:", all_data)


def test_data_transformer():
    """
    常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的。
    :return:
    """
    # 多项式转换，参数degree为度，默认值为2
    polynomial_data = PolynomialFeatures().fit_transform(iris.data)
    log_print_value("polynomial_data:", polynomial_data)

    # 自定义转换函数为对数函数的数据变换, 第一个参数是单变元函数
    function_transform_data = FunctionTransformer(log1p).fit_transform(iris.data)
    log_print_value("function_transform_data:", function_transform_data)


def mic(x, y):
    """
    mic
    :param x:
    :param y:
    :return:
    """
    m = MINE()
    m.compute_score(x, y)
    return m.mic()


def test_filter():
    """
    测试特征选择的过滤法。
    Filter, 按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
    :return:
    """
    # 方差选择法，返回值为特征选择后的数据，参数threshold为方差的阈值
    variance_filter_data = VarianceThreshold(threshold=3).fit_transform(iris.data)
    log_print_value("variance_filter_data:", variance_filter_data)

    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，
    # 数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # 参数k为选择的特征个数
    pearson_k_best_data = SelectKBest(lambda x_mat, y_mat: array(
        list(map(lambda x: pearsonr(x, y_mat), x_mat.T))).T[0], k=2)\
        .fit_transform(iris.data, iris.target)
    # func_pearson = lambda x_mat, y_mat: array(
    #     list(map(lambda x: pearsonr(x, y_mat), x_mat.T))).T
    # result = func_pearson(iris.data, iris.target)
    # log_print_value("result:", result)
    log_print_value("pearson_k_best_data:", pearson_k_best_data)

    # 经典的卡方检验是检验定性自变量对定性因变量的相关性。假设自变量有N种取值，因变量有M种取值，
    # 考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量。
    chi2_k_best_data = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
    log_print_value("chi2_k_best_data:", chi2_k_best_data)

    # 经典的互信息也是评价定性自变量对定性因变量的相关性的
    mic_k_best_data = SelectKBest(lambda x_mat, y_mat: array(list(map(lambda x: mic(x, y_mat), x_mat.T))).T, k=2)\
        .fit_transform(iris.data, iris.target)
    log_print_value("mic_k_best_data:", mic_k_best_data)


def test_wrapper():
    """
    测试特征选择的包装法。
    Wrapper, 根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
    :return:
    """
    # 递归特征消除法，返回特征选择后的数据，recursive feature elimination
    # 参数estimator为基模型
    # 参数n_features_to_select为选择的特征个数
    feature_data = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
    log_print_value("feature_data:", feature_data)


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight, random_state=random_state, solver=solver,
                                    max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                                    warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cnt_of_row, cnt_of_col = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cnt_of_row):
            for j in range(cnt_of_col):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cnt_of_col):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


def test_embedded():
    """
    测试特征选择的嵌入法。
    Embedded, 先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。
    :return:
    """
    # 带L1惩罚项的逻辑回归作为基模型的特征选择
    feature_data = SelectFromModel(LogisticRegression(penalty="l1", C=0.1))\
        .fit_transform(iris.data, iris.target)
    log_print_value("feature_data:", feature_data)

    # 带L1和L2惩罚项的逻辑回归作为基模型的特征选择，参数threshold为权值系数之差的阈值
    feature_data = SelectFromModel(LR(threshold=0.5, C=0.1))\
        .fit_transform(iris.data, iris.target)
    log_print_value("feature_data:", feature_data)

    # GBDT作为基模型的特征选择
    feature_data = SelectFromModel(GradientBoostingClassifier())\
        .fit_transform(iris.data, iris.target)
    log_print_value("feature_data:", feature_data)


def test_reduce_dimension():
    """
    测试特征的降维。
    :return:
    """
    # 主成分分析法，返回降维后的数据。参数n_components为主成分数目。
    feature_data = PCA(n_components=2).fit_transform(iris.data)
    log_print_value("feature_data:", feature_data)

    # 线性判别分析法，返回降维后的数据。参数n_components为降维后的维数。
    feature_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(iris.data, iris.target)
    log_print_value("feature_data:", feature_data)


if __name__ == "__main__":
    # show_data()
    # test_nondimensionalization()
    # test_binarize()
    # test_one_hot()
    # test_missing_value()
    # test_data_transformer()
    # test_filter()
    # test_wrapper()
    # test_embedded()
    test_reduce_dimension()
    pass


