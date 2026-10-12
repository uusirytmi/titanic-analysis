import unittest
import numpy as np
import pandas as pd
from experiments.next_round import Companions, group_ids, select_weights, WEIGHTS


def passengers():
    return pd.DataFrame(dict(PassengerId=[1,2,3,4], Name=['A, Mr. X','B, Mrs. Y','B, Mr. Z','D, Miss. W'],
        Ticket=['T 1','T1','T2','T3'], Cabin=['C1',None,'B2',None], SibSp=[1,1,1,0], Parch=[0]*4,
        Pclass=[1,2,2,3], Sex=['male','female','male','female'], Age=[20.,30.,40.,np.nan],
        Fare=[10.,20.,30.,40.], Embarked=['S']*4))


class NextRoundSafety(unittest.TestCase):
    def test_connected_components_close_transitively(self):
        X=passengers(); g=group_ids(X)
        self.assertEqual(g[0],g[1])  # shared normalized ticket
        self.assertEqual(g[1],g[2])  # shared family
        self.assertNotEqual(g[2],g[3])
        X['Survived']=[1,0,1,0]
        np.testing.assert_array_equal(g,group_ids(X))
        X['Survived']=1-X.Survived
        np.testing.assert_array_equal(g,group_ids(X))

    def test_counts_are_training_only_leave_self_out_and_batch_invariant(self):
        X=passengers(); f=Companions('family').fit(X.iloc[:2])
        np.testing.assert_array_equal(f.transform(X.iloc[:2]).TicketCompanions,[1,1])
        held=X.iloc[[2]].copy()
        self.assertEqual(f.transform(held).FamilyCompanions.iloc[0],1)
        self.assertEqual(f.transform(held).TicketCompanions.iloc[0],0)
        duplicates=pd.concat([held]*4,ignore_index=True)
        np.testing.assert_array_equal(f.transform(duplicates).FamilyCompanions,[1]*4)
        self.assertEqual(f.transform(X.iloc[[3]]).FamilyCompanions.iloc[0],0)
        a=Companions('family').fit(X,[0,1,0,1]).transform(X)
        b=Companions('family').fit(X,[1,0,1,0]).transform(X)
        pd.testing.assert_frame_equal(a,b)

    def test_unseen_group_has_no_training_companions(self):
        X=passengers(); X.loc[3,'Age']=25.; f=Companions('interactions').fit(X.iloc[[3]])
        z=f.transform(X.iloc[:3])
        self.assertTrue((z[['FamilyCompanions','TicketCompanions']]==0).all().all())

    def test_weights_use_only_supplied_inner_oof_and_ties_are_fixed(self):
        y=np.array([0,1]*3); folds=np.repeat([0,1,2],2)
        good=np.where(y, .9,.1); bad=1-good
        selected,losses=select_weights(np.column_stack([good,bad,bad]),y,folds)
        self.assertEqual(selected,1)
        self.assertEqual(selected,int(np.argmin(losses)))
        selected,_=select_weights(np.column_stack([good]*3),y,folds)
        self.assertEqual(selected,0)
        self.assertTrue(all(np.isclose(sum(w),1) for w in WEIGHTS))


if __name__=='__main__':
    unittest.main()
